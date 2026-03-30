from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import json, asyncio, networkx as nx
from tournament.arena import Arena
from graph.generator import Generator
from graph.architecture import Architecture
from backend.fight_viz import run_fight_visualization
from backend.fight_viz import run_tournament_fight
import glob
import os
import uuid
import os
import pickle
import math

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

arena = Arena(architecture_size=12, verbose=False)
generator = Generator(generation_type="agnostic")


def compute_dag_layout(arch, x_spacing=160, y_spacing=90):
    """Layered DAG layout: sources at top, sinks at bottom."""
    topo = list(nx.topological_sort(arch))

    # Assign depth = longest path from any source
    depth = {}
    for node in topo:
        preds = list(arch.predecessors(node))
        if not preds:
            depth[node] = 0
        else:
            depth[node] = max(depth[p] for p in preds) + 1

    # Group nodes by layer
    layers: dict[int, list] = {}
    for node, d in depth.items():
        layers.setdefault(d, []).append(node)

    # Position each node: centered horizontally within its layer
    pos = {}
    for d, layer_nodes in layers.items():
        n = len(layer_nodes)
        total_width = (n - 1) * x_spacing
        start_x = -total_width / 2
        for i, node in enumerate(layer_nodes):
            pos[node] = (start_x + i * x_spacing, d * y_spacing)

    return pos


# Temporary cache for architectures currently displayed on the frontend
session_archs = {}

@app.get("/api/generate")
def generate_architecture():
    arch = generator.generate(12)
    pos = compute_dag_layout(arch)

    # Generate a unique ID and cache the architecture
    arch_id = str(uuid.uuid4())
    session_archs[arch_id] = arch

    nodes = []
    for n in arch.nodes:
        mod = arch.nodes[n]['module']
        x, y = pos.get(n, (0, 0))
        nodes.append({
            "id": str(n),
            "type": mod.__class__.__name__,
            "module_type": mod.module_type.name,
            "x": x + 500,
            "y": y + 60,
        })
    edges = [{"source": str(u), "target": str(v)} for u, v in arch.edges]
    
    # Return the arch_id so the frontend can reference it later
    return {"arch_id": arch_id, "nodes": nodes, "edges": edges}



@app.websocket("/ws/tournament")
async def tournament_ws(websocket: WebSocket):
    await websocket.accept()
    data = await websocket.receive_json()

    n_random     = data.get("n_random", 8)
    loaded_files = data.get("loaded_archs", [])

    # ── build architecture pool ──
    architectures = []
    arch_info     = []

    for i in range(n_random):
        arch    = generator.generate(12)
        arch_id = str(uuid.uuid4())
        session_archs[arch_id] = arch
        architectures.append(arch)
        arch_info.append({
            "id": len(arch_info), "arch_id": arch_id,
            "name": f"Random {i}", "source": "random",
        })

    for filename in loaded_files:
        if os.path.exists(filename):
            arch    = Architecture.load(filename)
            arch_id = str(uuid.uuid4())
            session_archs[arch_id] = arch
            architectures.append(arch)
            arch_info.append({
                "id": len(arch_info), "arch_id": arch_id,
                "name": filename.replace(".pkl", ""), "source": "loaded",
            })

    n_archs = len(architectures)
    total   = n_archs * (n_archs - 1) // 2

    await websocket.send_json({
        "type": "init",
        "architectures": arch_info,
        "n_archs": n_archs,
        "total_fights": total,
    })

    if n_archs < 2:
        await websocket.send_json({"type": "done", "final_scores": []})
        return

    # ── accumulators (raw log-values, summed across fights) ──
    raw_learn_sum = [0.0] * n_archs
    raw_speed_sum = [0.0] * n_archs
    raw_time_sum  = [0.0] * n_archs
    fight_counts  = [0]   * n_archs

    fight = 0

    for i in range(n_archs):
        for j in range(i + 1, n_archs):
            fight += 1
            failed = False

            try:
                result = await asyncio.to_thread(
                    arena.get_scores,
                    architectures[i], architectures[j],
                    randomizeHP=True, pcp=0, get_delays=True,
                )
                if result is None:
                    raise ValueError("get_scores returned None")
                score_i, score_j, delay_i, delay_j = result
            except Exception as e:
                print(f"Fight {i} vs {j} errored: {e}, using fallback scores")
                score_i, score_j = 1e-5, 1e-5
                delay_i, delay_j = 10.0, 10.0
                failed = True

            # ── raw log values (same formula as occam_selection) ──
            log_learn_i = math.log(max(score_i, 1e-10))
            log_learn_j = math.log(max(score_j, 1e-10))
            log_speed_i = math.log(max(1.0 / max(delay_i, 1e-6), 1e-10))
            log_speed_j = math.log(max(1.0 / max(delay_j, 1e-6), 1e-10))

            raw_learn_sum[i] += log_learn_i
            raw_learn_sum[j] += log_learn_j
            raw_speed_sum[i] += log_speed_i
            raw_speed_sum[j] += log_speed_j
            raw_time_sum[i]  += delay_i
            raw_time_sum[j]  += delay_j
            fight_counts[i]  += 1
            fight_counts[j]  += 1

            # ── current normalized metrics for ALL archs (leaderboard) ──
            scores_arr = []
            learns_arr = []
            speeds_arr = []
            times_arr  = []

            for k in range(n_archs):
                fc    = max(fight_counts[k], 1)
                avg_l = raw_learn_sum[k] / fc
                avg_s = raw_speed_sum[k] / fc
                nl    = (avg_l - arena.avg_learn) / arena.std_learn
                ns    = (avg_s - arena.avg_speed) / arena.std_speed
                comb  = nl * (1.0 - arena.speed_bal) + ns * arena.speed_bal
                learns_arr.append(nl)
                speeds_arr.append(ns)
                scores_arr.append(comb)
                times_arr.append(raw_time_sum[k] / fc)

            # ── per-fight normalized values (for fight log detail) ──
            fl_i = (log_learn_i - arena.avg_learn) / arena.std_learn
            fl_j = (log_learn_j - arena.avg_learn) / arena.std_learn
            fs_i = (log_speed_i - arena.avg_speed) / arena.std_speed
            fs_j = (log_speed_j - arena.avg_speed) / arena.std_speed
            fc_i = fl_i * (1.0 - arena.speed_bal) + fs_i * arena.speed_bal
            fc_j = fl_j * (1.0 - arena.speed_bal) + fs_j * arena.speed_bal

            await websocket.send_json({
                "type": "fight_result",
                "fight": fight, "total": total,
                "i": i, "j": j,
                "failed":  failed,
                "score_i": float(fc_i),  "score_j": float(fc_j),
                "learn_i": float(fl_i),  "learn_j": float(fl_j),
                "speed_i": float(fs_i),  "speed_j": float(fs_j),
                "time_i":  float(delay_i), "time_j":  float(delay_j),
                "loss_i":  float(1.0 / max(score_i ** 2, 1e-20)),
                "loss_j":  float(1.0 / max(score_j ** 2, 1e-20)),
                "scores":         [float(s) for s in scores_arr],
                "learnabilities": [float(s) for s in learns_arr],
                "speeds":         [float(s) for s in speeds_arr],
                "fit_times":      [float(t) for t in times_arr],
                "fight_counts":   fight_counts,
            })

    # ── final ──
    final_scores = []
    for k in range(n_archs):
        fc    = max(fight_counts[k], 1)
        avg_l = raw_learn_sum[k] / fc
        avg_s = raw_speed_sum[k] / fc
        nl    = (avg_l - arena.avg_learn) / arena.std_learn
        ns    = (avg_s - arena.avg_speed) / arena.std_speed
        final_scores.append(nl * (1.0 - arena.speed_bal) + ns * arena.speed_bal)

    await websocket.send_json({
        "type": "done",
        "final_scores": [float(s) for s in final_scores],
    })

@app.get("/api/fight_viz")
async def fight_viz(arch_a_file: str = None, arch_b_file: str = None):
    # Keep trying until we successfully simulate a valid fight
    generating_A = True
    generating_B = True
    if arch_a_file:
        generating_A = False
        if not os.path.exists(arch_a_file):
            raise HTTPException(status_code=404, detail=f"{arch_a_file} not found")
        arch_a = Architecture.load(arch_a_file)
    if arch_b_file:
        generating_B = False
        if not os.path.exists(arch_b_file):
            raise HTTPException(status_code=404, detail=f"{arch_b_file} not found")
        arch_b = Architecture.load(arch_b_file)


    attempts = 0
    while attempts < 300:
        try:
            if arch_a_file is None:
                arch_a = generator.generate(12)
            if arch_b_file is None:
                arch_b = generator.generate(12)

            arch_a_id = str(uuid.uuid4())
            arch_b_id = str(uuid.uuid4())
            session_archs[arch_a_id] = arch_a
            session_archs[arch_b_id] = arch_b
            
            result = await asyncio.to_thread(
                run_fight_visualization, arch_a, arch_b, max_iter=500, lr=5e-3, n_snapshots=50, generating_A=generating_A, generating_B=generating_B
            )
            result["fight_a"]["arch_id"] = arch_a_id  # A learning B (this is Arch A)
            result["fight_b"]["arch_id"] = arch_b_id  # B learning A (this is Arch B)
            return result
        except Exception as e:
            # If the architectures were completely broken, catch the error 
            # and let the loop generate a new pair automatically
            print(f"Skipping bad architecture pair: {e}")
            continue
    
@app.get("/api/saved_archs")
def list_saved_archs():
    # Find all .pkl files in the root directory
    files = glob.glob("*.pkl")
    return {"files": files}

@app.post("/api/save_arch")
def save_arch(arch_id: str, filename: str):
    if arch_id not in session_archs:
        return {"error": "Architecture not found in session cache. Try regenerating."}
    
    arch = session_archs[arch_id]
    
    # Ensure filename ends with .pkl
    if not filename.endswith(".pkl"):
        filename += ".pkl"
        
    try:
        arch.save(filename)
        return {"success": True, "message": f"Saved as {filename}"}
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/load_arch/{filename}")
def load_architecture(filename: str):
    if not os.path.exists(filename):
        return {"error": "File not found"}
        
    arch = Architecture.load(filename)
    pos = compute_dag_layout(arch)

    nodes = []
    for n in arch.nodes:
        mod = arch.nodes[n]['module']
        x, y = pos.get(n, (0, 0))
        nodes.append({
            "id": str(n),
            "type": mod.__class__.__name__,
            "module_type": mod.module_type.name,
            "x": x + 500,
            "y": y + 60,
        })

    edges = [{"source": str(u), "target": str(v)} for u, v in arch.edges]
    return {"nodes": nodes, "edges": edges}

@app.get("/")
def root():
    return {
        "status": "Emernet API running",
        "endpoints": ["/api/generate", "/ws/tournament", "/docs"],
    }