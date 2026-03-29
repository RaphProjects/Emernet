from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import json, asyncio, networkx as nx
from tournament.arena import Arena
from graph.generator import Generator
from graph.architecture import Architecture
from backend.fight_viz import run_fight_visualization
import glob
import os
import uuid
import os
import pickle

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
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
    n_archs = data.get("n_archs", 8)

    architectures = [generator.generate(12) for _ in range(n_archs)]
    scores = [0.0] * n_archs
    total = n_archs * (n_archs - 1) // 2
    fight = 0

    for i in range(n_archs):
        for j in range(i + 1, n_archs):
            fight += 1

            result = await asyncio.to_thread(
                arena.get_scores,
                architectures[i], architectures[j], pcp=0
            )

            if result is None:
                continue

            s_i, s_j = result
            scores[i] += s_i
            scores[j] += s_j

            await websocket.send_json({
                "type": "fight_result",
                "fight": fight,
                "total": total,
                "i": i, "j": j,
                "score_i": float(s_i),
                "score_j": float(s_j),
                "leaderboard": [float(s) for s in scores],
            })

    await websocket.send_json({
        "type": "done",
        "final_scores": [float(s) for s in scores],
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