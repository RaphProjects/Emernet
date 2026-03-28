from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import json, asyncio, networkx as nx
from tournament.arena import Arena
from graph.generator import Generator
from graph.architecture import Architecture

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


@app.get("/api/generate")
def generate_architecture():
    arch = generator.generate(12)
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


@app.get("/")
def root():
    return {
        "status": "Emernet API running",
        "endpoints": ["/api/generate", "/ws/tournament", "/docs"],
    }