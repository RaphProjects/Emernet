from fastapi import FastAPI, HTTPException, WebSocket, Query, Form, Request
from starlette.websockets import WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import json, asyncio, networkx as nx
from tournament.arena import (
    Arena,
    get_normalization_values,
    set_normalization_values,
    normalization_key_for_modules,
    get_speed_balance_values,
    set_speed_balance_values,
)
from graph.generator import Generator
from graph.architecture import Architecture
from graph.mutator import Mutator
from graph.validation import architecture_execution_errors
from backend.model_exporter import export_architecture_python
from backend.fight_viz import run_fight_visualization
from backend.fight_viz import run_tournament_fight
import pickle, io, uuid, time
from fastapi import UploadFile, File
from fastapi.responses import StreamingResponse
import glob
import os
import math
import copy
import torch
import numpy as np
from modules.activations import Activation

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
arch_store: dict[str, "Architecture"] = {}
gnn_store: dict[str, dict] = {}
lgbm_store: dict[str, object] = {}
policy_store: dict[str, dict] = {}
arena = Arena(architecture_size=12, verbose=False)
generator = Generator(generation_type="agnostic")
CLIENT_SETTINGS_FILE = "arena_client_settings.json"


class _NumpyCompatibleUnpickler(pickle.Unpickler):
    """Load NumPy 2 pickles on deployments that still use NumPy 1.x."""

    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = "numpy.core" + module[len("numpy._core"):]
        return super().find_class(module, name)


def _pickle_loads_compatible(contents):
    return _NumpyCompatibleUnpickler(io.BytesIO(contents)).load()


def _load_client_settings():
    if not os.path.exists(CLIENT_SETTINGS_FILE):
        return {}
    try:
        with open(CLIENT_SETTINGS_FILE, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        return loaded if isinstance(loaded, dict) else {}
    except Exception:
        return {}


CLIENT_SETTINGS = _load_client_settings()


def _save_client_settings():
    with open(CLIENT_SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(CLIENT_SETTINGS, f, indent=2)


def _client_id_from_request(request: Request):
    return request.headers.get("x-emernet-client-id") or request.query_params.get("client_id")


def _normalization_from_payload(values, base=None):
    base_values = copy.deepcopy(base or get_normalization_values())
    if "default" in values and "Rich" not in values:
        values = {**values, "Rich": values["default"]}
    for key in ("Unified", "Rich", "All"):
        if key not in values:
            continue
        base_values.setdefault(key, {})
        for field in ("avg_learn", "std_learn", "avg_speed", "std_speed", "avg_simp", "std_simp"):
            if field in values[key]:
                base_values[key][field] = float(values[key][field])
        if base_values[key]["std_learn"] <= 0 or base_values[key]["std_speed"] <= 0 or base_values[key]["std_simp"] <= 0:
            raise ValueError("Normalization std values must be positive")
    return base_values


def _speed_balance_from_payload(values, base=None):
    base_values = copy.deepcopy(base or get_speed_balance_values())
    if "default" in values and "Rich" not in values:
        values = {**values, "Rich": values["default"]}
    for key in ("Unified", "Rich", "All"):
        if key not in values:
            continue
        value = float(values[key])
        if value < 0.0 or value > 1.0:
            raise ValueError("speed_bal must be between 0 and 1")
        base_values[key] = value
    return base_values


def _client_normalization(client_id=None):
    if client_id and client_id in CLIENT_SETTINGS and "normalization" in CLIENT_SETTINGS[client_id]:
        return _normalization_from_payload(CLIENT_SETTINGS[client_id]["normalization"])
    return get_normalization_values()


def _client_speed_balance(client_id=None):
    if client_id and client_id in CLIENT_SETTINGS and "speed_balance" in CLIENT_SETTINGS[client_id]:
        return _speed_balance_from_payload(CLIENT_SETTINGS[client_id]["speed_balance"])
    return get_speed_balance_values()


def _set_client_normalization(client_id, values):
    normalized = _normalization_from_payload(values, _client_normalization(client_id))
    CLIENT_SETTINGS.setdefault(client_id, {})["normalization"] = normalized
    _save_client_settings()
    return normalized


def _set_client_speed_balance(client_id, values):
    speed_balance = _speed_balance_from_payload(values, _client_speed_balance(client_id))
    CLIENT_SETTINGS.setdefault(client_id, {})["speed_balance"] = speed_balance
    _save_client_settings()
    return speed_balance


def _apply_client_arena_settings(arena_obj, module_set="Unified", client_id=None):
    group = normalization_key_for_modules(module_set)
    norm = _client_normalization(client_id)[group]
    speeds = _client_speed_balance(client_id)
    arena_obj.avg_learn = norm["avg_learn"]
    arena_obj.std_learn = norm["std_learn"]
    arena_obj.avg_speed = norm["avg_speed"]
    arena_obj.std_speed = norm["std_speed"]
    arena_obj.avg_simp = norm.get("avg_simp", norm["avg_learn"])
    arena_obj.std_simp = norm.get("std_simp", norm["std_learn"])
    arena_obj.speed_bal = speeds[group]
    return arena_obj


def _make_arena(module_set="Unified", client_id=None, **kwargs):
    return _apply_client_arena_settings(Arena(modules=module_set, **kwargs), module_set, client_id)


def compute_dag_layout(arch, x_spacing=160, y_spacing=90):
    """Layered DAG layout: sources at top, sinks at bottom."""
    topo = list(nx.topological_sort(arch))

    depth = {}
    for node in topo:
        preds = list(arch.predecessors(node))
        if not preds:
            depth[node] = 0
        else:
            depth[node] = max(depth[p] for p in preds) + 1

    layers: dict[int, list] = {}
    for node, d in depth.items():
        layers.setdefault(d, []).append(node)

    pos = {}
    for d, layer_nodes in layers.items():
        n = len(layer_nodes)
        total_width = (n - 1) * x_spacing
        start_x = -total_width / 2
        for i, node in enumerate(layer_nodes):
            pos[node] = (start_x + i * x_spacing, d * y_spacing)

    return pos


def arch_to_graph_data(arch, arch_id=None):
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
    result = {"nodes": nodes, "edges": edges}
    if arch_id:
        result["arch_id"] = arch_id
    return result


def _node_label_for_error(arch, node_id):
    if node_id not in arch.nodes:
        return str(node_id)
    module = arch.nodes[node_id].get("module")
    module_name = module.__class__.__name__ if module is not None else "unknown"
    return f"{node_id} ({module_name})"


def _cycle_label_for_error(arch):
    try:
        cycle = nx.find_cycle(arch)
    except Exception:
        return "unknown cycle"
    if not cycle:
        return "unknown cycle"
    cycle_nodes = [edge[0] for edge in cycle] + [cycle[0][0]]
    return " -> ".join(_node_label_for_error(arch, node_id) for node_id in cycle_nodes)


def _logit_from_unit_interval(value):
    v = max(1e-6, min(1.0 - 1e-6, float(value)))
    return math.log(v / (1.0 - v))


def _set_module_param(module, key, value):
    if isinstance(module, Activation) and key in {"sharpness", "symmetry", "gate"}:
        if module.learnable:
            raw_name = f"raw_{key}"
            raw = getattr(module, raw_name, None)
            if isinstance(raw, torch.nn.Parameter):
                with torch.no_grad():
                    raw.copy_(torch.tensor(_logit_from_unit_interval(value), dtype=raw.dtype, device=raw.device))
            return
        current = getattr(module, key, None)
        if isinstance(current, torch.Tensor):
            current.copy_(torch.tensor(float(value), dtype=current.dtype, device=current.device))
        else:
            setattr(module, key, torch.tensor(float(value)))
        return
    if hasattr(module, key):
        current = getattr(module, key)
        if isinstance(current, torch.Tensor):
            current.copy_(torch.tensor(value, dtype=current.dtype, device=current.device))
        else:
            setattr(module, key, value)


def _module_params_for_ui(module):
    params = {}
    if isinstance(module, Activation):
        if module.learnable:
            params["sharpness"] = float(torch.sigmoid(module.raw_sharpness).detach().cpu().item())
            params["symmetry"] = float(torch.sigmoid(module.raw_symmetry).detach().cpu().item())
            params["gate"] = float(torch.sigmoid(module.raw_gate).detach().cpu().item())
        else:
            params["sharpness"] = float(module.sharpness.detach().cpu().item())
            params["symmetry"] = float(module.symmetry.detach().cpu().item())
            params["gate"] = float(module.gate.detach().cpu().item())
        params["learnable"] = bool(module.learnable)
        return params
    for key, value in module.__dict__.items():
        if not key.startswith('_') and not callable(value):
            if isinstance(value, (int, float, str, bool, list, tuple)):
                params[key] = value
            elif isinstance(value, torch.Tensor):
                params[key] = value.tolist()
    return params


@app.post("/api/upload_arch")
async def upload_arch(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        arch = _pickle_loads_compatible(contents)

        arch_id = str(uuid.uuid4())
        arch_store[arch_id] = arch

        return arch_to_graph_data(arch, arch_id=arch_id)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

@app.post("/api/import_arch_subgraph")
async def import_arch_subgraph(arch_id: str = Form(...), file: UploadFile = File(...)):
    try:
        contents = await file.read()
        sub_arch = _pickle_loads_compatible(contents)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid .pkl file: {e}")

    if arch_id not in arch_store:
        raise HTTPException(status_code=404, detail="Architecture not found")

    arch = copy.deepcopy(arch_store[arch_id])

    try:
        topo = list(nx.topological_sort(sub_arch))
    except nx.NetworkXUnfeasible:
        raise HTTPException(status_code=400, detail="Sub-architecture contains a cycle")

    id_map = {}
    for old_id in topo:
        module = copy.deepcopy(sub_arch.nodes[old_id]['module'])
        new_id = arch.append_node(module)
        id_map[old_id] = new_id

    for old_u, old_v in sub_arch.edges():
        arch.add_edge(id_map[old_u], id_map[old_v])

    for old_id in topo:
        module = sub_arch.nodes[old_id]['module']
        if module.mapping_type.name == "SOURCE":
            new_id = id_map[old_id]
            if len(list(arch.predecessors(new_id))) == 0:
                arch.add_edge(0, new_id)

    if not arch.isValid():
        raise HTTPException(status_code=400, detail="Resulting architecture is invalid after import")

    new_arch_id = str(uuid.uuid4())
    arch_store[new_arch_id] = arch
    return arch_to_graph_data(arch, arch_id=new_arch_id)

@app.get("/api/download_arch/{arch_id}")
async def download_arch(arch_id: str):
    arch = arch_store.get(arch_id)
    if arch is None:
        return {"error": "Architecture not found"}

    buf = io.BytesIO()
    pickle.dump(arch, buf)
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f"attachment; filename={arch_id}.pkl"},
    )


@app.get("/api/download_arch_python/{arch_id}")
async def download_arch_python(arch_id: str):
    arch = arch_store.get(arch_id)
    if arch is None:
        raise HTTPException(status_code=404, detail="Architecture not found")
    try:
        source = export_architecture_python(arch)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    buf = io.BytesIO(source.encode("utf-8"))
    return StreamingResponse(
        buf,
        media_type="text/x-python",
        headers={"Content-Disposition": f"attachment; filename=emernet_arch_{arch_id}.py"},
    )


def _load_gnn_artifact(payload):
    from search.encoder import ArchEncoder, RewardPredictor, NODE_FEATURE_DIM
    from search.rl_search import GNN_EMBEDDING_DIM

    if isinstance(payload, dict) and "encoder_state" in payload:
        hidden_dim = int(payload.get("hidden_dim", 128))
        n_layers = int(payload.get("n_layers", 4))
        encoder = ArchEncoder(NODE_FEATURE_DIM, hidden_dim=hidden_dim, n_layers=n_layers)
        encoder.load_state_dict(payload["encoder_state"])
        predictor = None
        if payload.get("predictor_state") is not None:
            predictor = RewardPredictor(input_dim=int(payload.get("embedding_dim", GNN_EMBEDDING_DIM)))
            predictor.load_state_dict(payload["predictor_state"])
        return {"encoder": encoder, "predictor": predictor, "metadata": payload.get("metadata", {})}

    if isinstance(payload, dict) and "encoder" in payload:
        return {"encoder": payload["encoder"], "predictor": payload.get("predictor"), "metadata": payload.get("metadata", {})}

    raise ValueError("Invalid GNN artifact. Expected an encoder_state payload.")


def _dump_gnn_artifact(encoder, predictor=None, metadata=None):
    return {
        "kind": "emernet_gnn",
        "hidden_dim": getattr(encoder, "hidden_dim", 128),
        "n_layers": getattr(encoder, "n_layers", 4),
        "embedding_dim": getattr(encoder, "hidden_dim", 128) * 3,
        "encoder_state": encoder.cpu().state_dict(),
        "predictor_state": predictor.cpu().state_dict() if predictor is not None else None,
        "metadata": metadata or {},
    }


def _load_policy_artifact(payload):
    from search.evolution_policy import EvolutionPolicy
    if isinstance(payload, dict) and payload.get("kind") == "emernet_evolution_policy":
        gnn = _load_gnn_artifact(payload["gnn"])
        policy = payload.get("policy")
        if not isinstance(policy, EvolutionPolicy):
            raise ValueError("Policy payload does not contain an EvolutionPolicy")
        return {
            "encoder": gnn["encoder"],
            "predictor": gnn.get("predictor"),
            "policy": policy,
            "metadata": payload.get("metadata", {}),
        }
    if isinstance(payload, dict) and "encoder" in payload and "policy" in payload:
        return {
            "encoder": payload["encoder"],
            "predictor": payload.get("predictor"),
            "policy": payload["policy"],
            "metadata": payload.get("metadata", {}),
        }
    raise ValueError("Invalid Evolution Policy file")


def _dump_policy_artifact(encoder, predictor, policy, metadata=None):
    return {
        "kind": "emernet_evolution_policy",
        "version": 1,
        "gnn": _dump_gnn_artifact(encoder, predictor, metadata=metadata),
        "policy": policy,
        "metadata": metadata or {},
    }


@app.post("/api/upload_gnn")
async def upload_gnn(file: UploadFile = File(...)):
    try:
        artifact = _load_gnn_artifact(_pickle_loads_compatible(await file.read()))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid GNN file: {e}")
    model_id = str(uuid.uuid4())
    gnn_store[model_id] = artifact
    return {
        "gnn_id": model_id,
        "has_predictor": artifact.get("predictor") is not None,
        "metadata": artifact.get("metadata", {}),
    }


@app.post("/api/upload_policy")
async def upload_policy(file: UploadFile = File(...)):
    try:
        artifact = _load_policy_artifact(_pickle_loads_compatible(await file.read()))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid Evolution Policy file: {e}")
    model_id = str(uuid.uuid4())
    policy_store[model_id] = artifact
    policy = artifact["policy"]
    return {
        "policy_id": model_id,
        "records_seen": getattr(policy, "records_seen", 0),
        "replay_records": len(getattr(policy, "replay_y", []) or []),
        "has_lgbm": getattr(getattr(policy, "lgbm_policy", None), "model", None) is not None,
        "has_nn": bool(getattr(policy, "nn_trained", False)),
        "has_meta": bool(getattr(policy, "meta_trained", False)),
        "metadata": artifact.get("metadata", {}),
    }


@app.post("/api/upload_lgbm")
async def upload_lgbm(file: UploadFile = File(...)):
    try:
        model = _pickle_loads_compatible(await file.read())
        if isinstance(model, dict) and "policy" in model:
            model = model["policy"]
        if not hasattr(model, "extract_feature_vector") or not hasattr(model, "predict"):
            raise ValueError("Object does not look like an Emernet LGBM policy")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid LGBM file: {e}")
    model_id = str(uuid.uuid4())
    lgbm_store[model_id] = model
    return {
        "lgbm_id": model_id,
        "trained": getattr(model, "model", None) is not None,
    }


@app.get("/api/download_gnn/{gnn_id}")
async def download_gnn(gnn_id: str):
    artifact = gnn_store.get(gnn_id)
    if artifact is None:
        raise HTTPException(status_code=404, detail="GNN not found")
    buf = io.BytesIO()
    pickle.dump(_dump_gnn_artifact(artifact["encoder"], artifact.get("predictor"), artifact.get("metadata")), buf)
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f"attachment; filename=gnn_{gnn_id}.pkl"},
    )


@app.get("/api/download_lgbm/{lgbm_id}")
async def download_lgbm(lgbm_id: str):
    model = lgbm_store.get(lgbm_id)
    if model is None:
        raise HTTPException(status_code=404, detail="LGBM not found")
    buf = io.BytesIO()
    pickle.dump(model, buf)
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f"attachment; filename=lgbm_{lgbm_id}.pkl"},
    )


@app.get("/api/download_policy/{policy_id}")
async def download_policy(policy_id: str):
    artifact = policy_store.get(policy_id)
    if artifact is None:
        raise HTTPException(status_code=404, detail="Evolution Policy not found")
    buf = io.BytesIO()
    pickle.dump(_dump_policy_artifact(artifact["encoder"], artifact.get("predictor"), artifact["policy"], artifact.get("metadata")), buf)
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f"attachment; filename=evolution_policy_{policy_id}.pkl"},
    )


@app.get("/api/arena_normalization")
def get_arena_normalization(request: Request):
    return {"normalization": _client_normalization(_client_id_from_request(request))}


@app.post("/api/arena_normalization")
def update_arena_normalization(payload: dict, request: Request):
    try:
        client_id = _client_id_from_request(request) or payload.get("client_id")
        incoming = payload.get("normalization", payload)
        normalization = _set_client_normalization(client_id, incoming) if client_id else set_normalization_values(incoming)
        norm = normalization["Unified"]
        arena.avg_learn = norm["avg_learn"]
        arena.std_learn = norm["std_learn"]
        arena.avg_speed = norm["avg_speed"]
        arena.std_speed = norm["std_speed"]
        arena.avg_simp = norm.get("avg_simp", arena.avg_learn)
        arena.std_simp = norm.get("std_simp", arena.std_learn)
        return {"normalization": normalization}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/speed_balance")
def get_speed_balance(request: Request):
    return {"speed_balance": _client_speed_balance(_client_id_from_request(request))}


@app.post("/api/speed_balance")
def update_speed_balance(payload: dict, request: Request):
    try:
        client_id = _client_id_from_request(request) or payload.get("client_id")
        incoming = payload.get("speed_balance", payload)
        values = _set_client_speed_balance(client_id, incoming) if client_id else set_speed_balance_values(incoming)
        arena.speed_bal = values["Unified"]
        return {"speed_balance": values}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.websocket("/ws/normalization_calibration")
async def normalization_calibration_ws(websocket: WebSocket):
    await websocket.accept()
    data = await websocket.receive_json()

    n_tournaments = max(1, int(data.get("n_tournaments", 3)))
    n_random = max(2, int(data.get("n_random", 8)))
    module_set = data.get("module_set", "Unified")
    client_id = data.get("client_id")
    arch_size = max(3, int(data.get("arch_size", 12)))
    max_duration = max(1, int(data.get("max_duration", 600)))
    dataset_size = max(16, int(data.get("dataset_size", 320)))
    n_fights = max(1, int(data.get("n_fights", 1)))

    calib_arena = _make_arena(
        n_fights=n_fights,
        dataset_size=dataset_size,
        architecture_size=arch_size,
        verbose=False,
        module_set=module_set,
        client_id=client_id,
    )
    gen = Generator(generation_type="agnostic", module_types=module_set)
    total_fights = n_tournaments * (n_random * (n_random - 1) // 2)
    fight_global = 0
    learn_samples = []
    speed_samples = []
    simp_samples = []

    await websocket.send_json({
        "type": "calibration_start",
        "n_tournaments": n_tournaments,
        "n_random": n_random,
        "total_fights": total_fights,
    })

    try:
        for tournament_idx in range(1, n_tournaments + 1):
            architectures = []
            for arch_idx in range(n_random):
                architectures.append(gen.generate(arch_size))
                await websocket.send_json({
                    "type": "generation_progress",
                    "tournament": tournament_idx,
                    "current": arch_idx + 1,
                    "total": n_random,
                })

            log_scores = [[0.0] * n_random for _ in range(n_random)]
            raw_learn_sum = [0.0] * n_random
            raw_speed_sum = [0.0] * n_random
            fight_counts = [0] * n_random

            for i in range(n_random):
                for j in range(i + 1, n_random):
                    fight_global += 1
                    failed = False
                    try:
                        score_i, score_j, delay_i, delay_j = calib_arena.get_scores(
                            architectures[i], architectures[j],
                            randomizeHP=True, pcp=0, get_delays=True,
                            max_duration=max_duration,
                        )
                    except Exception:
                        score_i, score_j = 1e-5, 1e-5
                        delay_i, delay_j = 10.0, 10.0
                        failed = True

                    log_learn_i = math.log(max(score_i, 1e-10))
                    log_learn_j = math.log(max(score_j, 1e-10))
                    log_speed_i = math.log(max(1.0 / max(delay_i, 1e-6), 1e-10))
                    log_speed_j = math.log(max(1.0 / max(delay_j, 1e-6), 1e-10))

                    log_scores[i][j] = log_learn_i
                    log_scores[j][i] = log_learn_j
                    raw_learn_sum[i] += log_learn_i
                    raw_learn_sum[j] += log_learn_j
                    raw_speed_sum[i] += log_speed_i
                    raw_speed_sum[j] += log_speed_j
                    fight_counts[i] += 1
                    fight_counts[j] += 1

                    await websocket.send_json({
                        "type": "calibration_fight",
                        "fight": fight_global,
                        "total_fights": total_fights,
                        "tournament": tournament_idx,
                        "failed": failed,
                    })

            tournament_learns = []
            tournament_speeds = []
            tournament_simps = []
            for idx in range(n_random):
                fc = max(fight_counts[idx], 1)
                learn_value = raw_learn_sum[idx] / fc
                speed_value = raw_speed_sum[idx] / fc
                vals = [
                    log_scores[row][idx]
                    for row in range(n_random)
                    if row != idx and log_scores[row][idx] != 0.0
                ]
                simp_value = sum(vals) / len(vals) if vals else learn_value
                learn_samples.append(learn_value)
                speed_samples.append(speed_value)
                simp_samples.append(simp_value)
                tournament_learns.append(learn_value)
                tournament_speeds.append(speed_value)
                tournament_simps.append(simp_value)

            await websocket.send_json({
                "type": "calibration_tournament_done",
                "tournament": tournament_idx,
                "learn_mean": float(np.mean(tournament_learns)),
                "speed_mean": float(np.mean(tournament_speeds)),
                "simp_mean": float(np.mean(tournament_simps)),
            })

        result = {
            "module_group": normalization_key_for_modules(module_set),
            "n_samples": len(learn_samples),
            "learnability": {
                "mean": float(np.mean(learn_samples)),
                "std": float(np.std(learn_samples) or 1e-6),
            },
            "speed": {
                "mean": float(np.mean(speed_samples)),
                "std": float(np.std(speed_samples) or 1e-6),
            },
            "opp_simp_raw": {
                "mean": float(np.mean(simp_samples)),
                "std": float(np.std(simp_samples) or 1e-6),
            },
        }
        await websocket.send_json({"type": "calibration_done", "result": result})
    except Exception as exc:
        await websocket.send_json({"type": "error", "message": str(exc)})
    finally:
        await websocket.close()


@app.websocket("/ws/speed_balance_calibration")
async def speed_balance_calibration_ws(websocket: WebSocket):
    await websocket.accept()
    data = await websocket.receive_json()

    n_tournaments = max(1, int(data.get("n_tournaments", 3)))
    pool_size = max(2, int(data.get("pool_size", 8)))
    module_set = data.get("module_set", "Unified")
    client_id = data.get("client_id")
    arch_size = max(3, int(data.get("arch_size", 12)))
    max_duration = max(1, int(data.get("max_duration", 600)))
    dataset_size = max(16, int(data.get("dataset_size", 320)))
    n_fights = max(1, int(data.get("n_fights", 1)))

    speed_arena = _make_arena(
        n_fights=n_fights,
        dataset_size=dataset_size,
        architecture_size=arch_size,
        verbose=False,
        module_set=module_set,
        client_id=client_id,
    )
    gen = Generator(generation_type="agnostic", module_types=module_set)
    mlp_dims = [[16, 16], [32, 64, 32, 32]]
    total_fights = n_tournaments * len(mlp_dims) * pool_size
    fight_global = 0
    speed_bals = []
    rounds = []

    await websocket.send_json({
        "type": "speed_balance_start",
        "n_tournaments": n_tournaments,
        "opponent_pool_size": pool_size,
        "mlp_count": len(mlp_dims),
        "total_fights": total_fights,
        "module_group": normalization_key_for_modules(module_set),
        "current_speed_bal": speed_arena.speed_bal,
    })

    try:
        for tournament_idx in range(1, n_tournaments + 1):
            opponents = []
            for opponent_idx in range(pool_size):
                opponents.append(gen.generate(arch_size))
                await websocket.send_json({
                    "type": "speed_balance_generation",
                    "tournament": tournament_idx,
                    "current": opponent_idx + 1,
                    "total": pool_size,
                })

            mlp_learn = []
            mlp_speed = []
            for mlp_idx, dims in enumerate(mlp_dims):
                mlp = speed_arena.make_mlp(hidden_sizes=dims)
                learn_sum = 0.0
                speed_sum = 0.0
                for opponent_idx, opponent in enumerate(opponents):
                    fight_global += 1
                    failed = False
                    try:
                        score_i, _, delay_i, _ = speed_arena.get_scores(
                            mlp, opponent,
                            randomizeHP=True, pcp=0, get_delays=True,
                            max_duration=max_duration,
                        )
                    except Exception:
                        score_i = 1e-5
                        delay_i = 10.0
                        failed = True

                    learn_sum += math.log(max(score_i, 1e-10))
                    speed_sum += math.log(max(1.0 / max(delay_i, 1e-6), 1e-10))

                    await websocket.send_json({
                        "type": "speed_balance_fight",
                        "fight": fight_global,
                        "total_fights": total_fights,
                        "tournament": tournament_idx,
                        "mlp_index": mlp_idx,
                        "mlp": "x".join(str(v) for v in dims),
                        "opponent": opponent_idx,
                        "failed": failed,
                    })

                raw_learn = learn_sum / max(pool_size, 1)
                raw_speed = speed_sum / max(pool_size, 1)
                mlp_learn.append((raw_learn - speed_arena.avg_learn) / speed_arena.std_learn)
                mlp_speed.append((raw_speed - speed_arena.avg_speed) / speed_arena.std_speed)

            mu_l = sum(mlp_learn) / len(mlp_learn)
            mu_s = sum(mlp_speed) / len(mlp_speed)
            num = 0.0
            den = 0.0
            for learn, speed in zip(mlp_learn, mlp_speed):
                delta_l = learn - mu_l
                delta_mix = (speed - mu_s) - delta_l
                num += delta_l * delta_mix
                den += delta_mix * delta_mix
            optimal = 0.5 if den == 0 else -num / den
            optimal = max(0.0, min(1.0, optimal))
            scores_at_optimal = [
                (learn * (1.0 - optimal)) + (speed * optimal)
                for learn, speed in zip(mlp_learn, mlp_speed)
            ]
            round_result = {
                "tournament": tournament_idx,
                "speed_bal": float(optimal),
                "score_spread": float(max(scores_at_optimal) - min(scores_at_optimal)),
                "mlps": [
                    {
                        "name": "x".join(str(v) for v in dims),
                        "learnability": float(learn),
                        "speed": float(speed),
                        "score": float(score),
                    }
                    for dims, learn, speed, score in zip(mlp_dims, mlp_learn, mlp_speed, scores_at_optimal)
                ],
            }
            speed_bals.append(optimal)
            rounds.append(round_result)
            await websocket.send_json({
                "type": "speed_balance_tournament_done",
                "result": round_result,
            })

        mean = float(np.mean(speed_bals))
        std = float(np.std(speed_bals))
        await websocket.send_json({
            "type": "speed_balance_done",
            "result": {
                "module_group": normalization_key_for_modules(module_set),
                "n_tournaments": n_tournaments,
                "speed_bal": mean,
                "std": std,
                "rounds": rounds,
            },
        })
    except Exception as exc:
        await websocket.send_json({"type": "error", "message": str(exc)})
    finally:
        await websocket.close()


REAL_DATASETS = [
    {
        "id": "california_housing",
        "label": "California Housing",
        "kind": "sklearn",
        "target_hint": "",
        "source_url": "https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html",
        "upload": "builtin",
    },
    {
        "id": "diabetes",
        "label": "Diabetes",
        "kind": "sklearn",
        "target_hint": "",
        "source_url": "https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html",
        "upload": "builtin",
    },
    {
        "id": "utkface_agedb",
        "label": "AgeDB / UTKFace",
        "kind": "image_folder",
        "target_hint": "Folder of images with age at the start of filename",
        "source_url": "https://www.kaggle.com/datasets/jangedoo/utkface-new",
        "upload": "folder",
    },
    {
        "id": "beijing_pm25",
        "label": "Beijing PM2.5 Air Quality",
        "kind": "csv",
        "target_hint": "CSV target column, usually pm2.5",
        "source_url": "https://www.kaggle.com/datasets/rupakroy/beijing-pm25-data-data-set",
        "upload": "csv_or_folder",
    },
    {
        "id": "electricity_load",
        "label": "Electricity Load Diagrams",
        "kind": "csv",
        "target_hint": "CSV target column or last numeric column",
        "source_url": "https://www.kaggle.com/datasets/uciml/electric-power-consumption-data-set",
        "upload": "csv_or_folder",
    },
]


@app.get("/api/real_datasets")
def list_real_datasets():
    return {"datasets": REAL_DATASETS}


def _split_scale_dataset(X, y, test_size=0.3, seed=42):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    sx, sy = StandardScaler(), StandardScaler()
    X_train = sx.fit_transform(X_train)
    X_test = sx.transform(X_test)
    y_train = sy.fit_transform(y_train)
    y_test = sy.transform(y_test)
    return {
        "train_input": torch.tensor(X_train, dtype=torch.float32).unsqueeze(1),
        "train_target": torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1, 1),
        "test_input": torch.tensor(X_test, dtype=torch.float32).unsqueeze(1),
        "test_target": torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1, 1),
    }


def _load_csv_regression(path, target_column=None):
    import csv as csv_module

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        dialect = csv_module.Sniffer().sniff(sample, delimiters=",;\t")
        reader = csv_module.DictReader(f, dialect=dialect)
        rows = list(reader)
    if not rows:
        raise ValueError("CSV has no rows")
    numeric_columns = []
    for col in rows[0].keys():
        ok = 0
        for row in rows[:200]:
            try:
                float(row[col])
                ok += 1
            except Exception:
                pass
        if ok >= max(1, min(len(rows[:200]), 20) // 2):
            numeric_columns.append(col)
    if not numeric_columns:
        raise ValueError("No numeric columns found in CSV")
    if target_column is None or target_column not in numeric_columns:
        preferred = ["pm2.5", "PM2.5", "pm25", "load", "Load", "target"]
        target_column = next((c for c in preferred if c in numeric_columns), numeric_columns[-1])
    feature_columns = [c for c in numeric_columns if c != target_column]
    if not feature_columns:
        raise ValueError("CSV needs at least one numeric feature column besides target")
    X, y = [], []
    for row in rows:
        try:
            X.append([float(row[c]) for c in feature_columns])
            y.append(float(row[target_column]))
        except Exception:
            continue
    if len(X) < 10:
        raise ValueError("Not enough numeric rows after cleaning")
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32), target_column


def _load_age_image_folder(path, max_samples=2000):
    import random
    import re
    from PIL import Image

    rng = random.Random(42)
    files = []
    seen = 0
    for root, _, names in os.walk(path):
        for name in names:
            if name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                match = re.match(r"(\d{1,3})", name)
                if match:
                    item = (os.path.join(root, name), int(match.group(1)))
                    seen += 1
                    if len(files) < max_samples:
                        files.append(item)
                    else:
                        replace_at = rng.randrange(seen)
                        if replace_at < max_samples:
                            files[replace_at] = item
    if not files:
        raise ValueError("No age-labelled image files found. UTKFace names usually start with age_...")
    rng.shuffle(files)
    X, y = [], []
    for filepath, age in files:
        try:
            image = Image.open(filepath).convert("L").resize((32, 32))
            X.append(np.asarray(image, dtype=np.float32) / 255.0)
            y.append(float(age))
        except Exception:
            continue
    if len(X) < 10:
        raise ValueError("Not enough readable age-labelled images")
    return np.stack(X), np.asarray(y, dtype=np.float32)


def _load_real_dataset(dataset_id, options):
    from sklearn.datasets import fetch_california_housing, load_diabetes

    if dataset_id == "california_housing":
        data = fetch_california_housing()
        return _split_scale_dataset(data.data, data.target, test_size=options.get("test_size", 0.3))
    if dataset_id == "diabetes":
        data = load_diabetes()
        return _split_scale_dataset(data.data, data.target, test_size=options.get("test_size", 0.3))
    paths = options.get("paths", {})
    targets = options.get("target_columns", {})
    path = paths.get(dataset_id)
    if not path:
        raise ValueError("This dataset needs a server-local path")
    if dataset_id == "utkface_agedb":
        X, y = _load_age_image_folder(path, max_samples=int(options.get("subsample", 2000)))
        return _split_scale_dataset(X.reshape(len(X), -1), y, test_size=options.get("test_size", 0.3))
    if dataset_id in {"beijing_pm25", "electricity_load"}:
        X, y, _ = _load_csv_regression(path, targets.get(dataset_id))
        return _split_scale_dataset(X, y, test_size=options.get("test_size", 0.3))
    raise ValueError(f"Unknown dataset: {dataset_id}")


def _safe_upload_path(root, relative_path, fallback_name):
    from pathlib import PurePosixPath

    rel = relative_path or fallback_name
    parts = [
        part
        for part in PurePosixPath(rel.replace("\\", "/")).parts
        if part not in ("", ".", "..") and not part.endswith(":")
    ]
    if not parts:
        parts = [fallback_name or "upload.bin"]
    return os.path.join(root, *parts)


def _find_first_csv(root):
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if filename.lower().endswith((".csv", ".txt")):
                return os.path.join(dirpath, filename)
    return None


def _evaluate_arch_on_dataset(arch, ds, max_iter=80, subsample=2000):
    from graph.executor import Executor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_batch_size = 2048 if device.type == "cuda" else 32
    train_input = ds["train_input"]
    train_target = ds["train_target"]
    test_input = ds["test_input"]
    test_target = ds["test_target"]
    if subsample and len(train_input) > subsample:
        idx = torch.randperm(len(train_input))[:subsample]
        train_input = train_input[idx]
        train_target = train_target[idx]
    train_input = train_input.to(device)
    train_target = train_target.to(device)
    test_input = test_input.to(device)
    test_target = test_target.to(device)
    arch_copy = copy.deepcopy(arch)
    arch_copy.reset_state()
    executor = Executor(arch_copy).to(device)
    executor.randomize_weights()
    start_fit = time.time()
    executor.fit(
        train_input,
        train_target,
        verbose=False,
        lr=0.001,
        max_iter=max_iter,
        batch_size=min(len(train_input), max_batch_size),
        patience=10,
        min_delta=1e-7,
        cpu=False,
    )
    fit_delay = time.time() - start_fit
    outputs = []
    start_test = time.time()
    with torch.no_grad():
        for i in range(0, len(test_input), max_batch_size):
            outputs.append(executor.forward(test_input[i:i + max_batch_size])[0].detach().cpu())
    pred = torch.cat(outputs, dim=0).to(device)
    test_delay = time.time() - start_test
    mse = torch.nn.functional.mse_loss(pred, test_target).item()
    mae = torch.mean(torch.abs(pred - test_target)).item()
    target_var = torch.var(test_target).item()
    r2 = 1.0 - (mse / target_var) if target_var > 1e-12 else float("nan")
    return {
        "mse": float(mse),
        "mae": float(mae),
        "r2": float(r2),
        "score": float(1.0 / (mse + 1e-8)) if math.isfinite(mse) else 0.0,
        "fit_delay": float(fit_delay),
        "test_delay": float(test_delay),
        "train_samples": int(len(train_input)),
        "test_samples": int(len(test_input)),
    }


@app.post("/api/real_dataset_test")
def real_dataset_test(payload: dict):
    arch_id = payload.get("arch_id")
    arch = arch_store.get(arch_id)
    if arch is None:
        raise HTTPException(status_code=404, detail="Architecture not found")
    dataset_ids = payload.get("datasets") or ["california_housing", "diabetes"]
    max_iter = max(1, int(payload.get("max_iter", 80)))
    subsample = max(16, int(payload.get("subsample", 2000)))
    options = {
        "paths": payload.get("paths", {}),
        "target_columns": payload.get("target_columns", {}),
        "test_size": float(payload.get("test_size", 0.3)),
        "subsample": subsample,
    }
    results = {}
    for dataset_id in dataset_ids:
        try:
            ds = _load_real_dataset(dataset_id, options)
            results[dataset_id] = {
                "status": "ok",
                **_evaluate_arch_on_dataset(arch, ds, max_iter=max_iter, subsample=subsample),
            }
        except Exception as e:
            results[dataset_id] = {"status": "error", "error": str(e)}
    return {"results": results}


@app.post("/api/real_dataset_upload_test")
async def real_dataset_upload_test(request: Request):
    import tempfile

    form = await request.form(max_files=50000, max_fields=20000)
    arch_id = str(form.get("arch_id", ""))
    datasets = str(form.get("datasets", "[]"))
    max_iter = int(form.get("max_iter", 80))
    subsample = int(form.get("subsample", 2000))
    test_size = float(form.get("test_size", 0.3))
    paths = form.get("paths", "{}")
    target_columns = form.get("target_columns", "{}")
    upload_manifest_raw = form.get("upload_manifest", "[]")
    files = form.getlist("files")
    legacy_file_datasets = form.getlist("file_datasets")
    legacy_relative_paths = form.getlist("relative_paths")

    arch = arch_store.get(arch_id)
    if arch is None:
        raise HTTPException(status_code=404, detail="Architecture not found")
    try:
        if not isinstance(paths, str):
            paths = "{}"
        if not isinstance(target_columns, str):
            target_columns = "{}"
        dataset_ids = json.loads(datasets)
        path_values = json.loads(paths or "{}")
        target_values = json.loads(target_columns or "{}")
        upload_manifest = json.loads(upload_manifest_raw or "[]") if isinstance(upload_manifest_raw, str) else []
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid request metadata: {e}")

    with tempfile.TemporaryDirectory(prefix="emernet_realdata_") as tmp:
        upload_roots = {}
        for idx, upload in enumerate(files):
            if not hasattr(upload, "filename") or not hasattr(upload, "read"):
                continue
            manifest_row = upload_manifest[idx] if idx < len(upload_manifest) and isinstance(upload_manifest[idx], dict) else {}
            dataset_id = manifest_row.get("dataset_id")
            if not dataset_id and idx < len(legacy_file_datasets):
                dataset_id = legacy_file_datasets[idx]
            if not dataset_id:
                continue
            root = upload_roots.setdefault(dataset_id, os.path.join(tmp, dataset_id))
            rel = manifest_row.get("relative_path") or (legacy_relative_paths[idx] if idx < len(legacy_relative_paths) else upload.filename)
            dest = _safe_upload_path(root, rel, upload.filename)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest, "wb") as f:
                f.write(await upload.read())

        for dataset_id, root in upload_roots.items():
            if dataset_id in {"beijing_pm25", "electricity_load"}:
                csv_path = _find_first_csv(root)
                if csv_path:
                    path_values[dataset_id] = csv_path
            else:
                path_values[dataset_id] = root

        payload = {
            "arch_id": arch_id,
            "datasets": dataset_ids,
            "max_iter": max_iter,
            "subsample": subsample,
            "test_size": test_size,
            "paths": path_values,
            "target_columns": target_values,
        }
        return real_dataset_test(payload)


def _policy_info(policy):
    replay_count = len(getattr(policy, "replay_y", []) or [])
    meta_weights = {}
    meta_layer = getattr(policy, "meta_layer", None)
    linear_layer = None
    if meta_layer is not None:
        if hasattr(meta_layer, "linear"):
            linear_layer = meta_layer.linear
        elif hasattr(meta_layer, "output"):
            linear_layer = meta_layer.output
    if linear_layer is not None:
        try:
            weight = linear_layer.weight.detach().cpu().numpy()
            bias = linear_layer.bias.detach().cpu().numpy()
            input_names = [f"meta_hidden_{idx}" for idx in range(weight.shape[1])] if hasattr(meta_layer, "output") and not hasattr(meta_layer, "linear") else ["lgbm_learnability", "lgbm_speed", "lgbm_opp_raw"]
            if not (hasattr(meta_layer, "output") and not hasattr(meta_layer, "linear")):
                latent_dim = max(0, weight.shape[1] - len(input_names)) if weight.ndim == 2 else 0
                input_names += [f"nn_latent_{idx}" for idx in range(latent_dim)]
            metric_names = ["learnability", "speed", "opp_raw"]
            rows = []
            if weight.ndim == 2:
                for metric_idx, metric in enumerate(metric_names[:weight.shape[0]]):
                    for input_idx, value in enumerate(weight[metric_idx]):
                        rows.append({
                            "metric": metric,
                            "name": input_names[input_idx] if input_idx < len(input_names) else f"feature_{input_idx}",
                            "value": float(value),
                        })
            else:
                rows = [{"metric": "final_score", "name": f"feature_{idx}", "value": float(value)} for idx, value in enumerate(weight.reshape(-1))]
            meta_weights = {
                "bias": [float(v) for v in np.asarray(bias).reshape(-1)],
                "weights": rows,
                "kind": "mlp" if hasattr(meta_layer, "output") and not hasattr(meta_layer, "linear") else "linear",
                "input_dim": getattr(meta_layer, "input_dim", None),
                "hidden_dim": getattr(meta_layer, "hidden_dim", None),
                "output_dim": getattr(meta_layer, "output_dim", None),
            }
        except Exception:
            meta_weights = {}
    return {
        "feature_dim": getattr(policy, "feature_dim", None),
        "records_seen": getattr(policy, "records_seen", replay_count),
        "replay_records": replay_count,
        "has_lgbm": getattr(getattr(policy, "lgbm_policy", None), "model", None) is not None,
        "has_nn": bool(getattr(policy, "nn_trained", False)),
        "has_meta": bool(getattr(policy, "meta_trained", False)),
        "meta": meta_weights,
        "metrics": [
            {
                "metric": "final_score",
                "lgbm": "derived from metrics",
                "nn": "derived from latent",
                "meta": "derived from metrics",
            },
            {
                "metric": "learnability",
                "lgbm": "trained" if getattr(getattr(policy, "lgbm_policy", None), "model", None) is not None else "not trained",
                "nn": "latent" if getattr(policy, "nn_trained", False) else "not trained",
                "meta": "trained" if getattr(policy, "meta_trained", False) else "not trained",
            },
            {
                "metric": "speed",
                "lgbm": "trained" if getattr(getattr(policy, "lgbm_policy", None), "model", None) is not None else "not trained",
                "nn": "latent" if getattr(policy, "nn_trained", False) else "not trained",
                "meta": "trained" if getattr(policy, "meta_trained", False) else "not trained",
            },
            {
                "metric": "opp_raw",
                "lgbm": "trained" if getattr(getattr(policy, "lgbm_policy", None), "model", None) is not None else "not trained",
                "nn": "latent" if getattr(policy, "nn_trained", False) else "not trained",
                "meta": "trained" if getattr(policy, "meta_trained", False) else "not trained",
            },
        ],
    }


@app.get("/api/policy_info/{policy_id}")
def policy_info(policy_id: str):
    artifact = policy_store.get(policy_id)
    if artifact is None:
        raise HTTPException(status_code=404, detail="Evolution Policy not found")
    return {
        "policy_id": policy_id,
        "metadata": artifact.get("metadata", {}),
        **_policy_info(artifact["policy"]),
    }


@app.post("/api/policy_predict_arch")
def policy_predict_arch(payload: dict):
    arch_id = payload.get("arch_id")
    policy_id = payload.get("policy_id")
    module_set = payload.get("module_set", "Unified")
    client_id = payload.get("client_id")
    speed_bal = float(payload.get("speed_bal", arena.speed_bal))
    opp_simp_bal = float(payload.get("opp_simp_bal", 0.0))

    arch = arch_store.get(arch_id)
    if arch is None:
        raise HTTPException(status_code=404, detail="Architecture not found")
    artifact = policy_store.get(policy_id)
    if artifact is None:
        raise HTTPException(status_code=404, detail="Evolution Policy not found")

    from search.rl_search import RLSearch

    eval_arena = _make_arena(architecture_size=max(1, len(arch.nodes)), verbose=False, module_set=module_set, client_id=client_id)
    searcher = RLSearch(arena=eval_arena, module_set=module_set, verbose=False)
    encoder = artifact["encoder"].to(searcher.device)
    policy = artifact["policy"]
    h_graph = searcher._get_embedding(encoder, arch)
    features = policy.lgbm_policy.extract_feature_vector(h_graph, {})
    comps = policy.predict_components(
        np.array([features], dtype=np.float32),
        speed_bal=speed_bal,
        opp_simp_bal=opp_simp_bal,
    )
    return {
        "learnability": float(comps["learnability"][0]),
        "speed": float(comps["speed"][0]),
        "opp_simp_raw": float(comps["opp_simp_raw"][0]),
        "final_score": float(comps["final"][0]),
        "speed_bal": speed_bal,
        "opp_simp_bal": opp_simp_bal,
        "records_seen": getattr(policy, "records_seen", None),
        "normalization_group": normalization_key_for_modules(module_set),
    }


def _policy_predict_for_architectures(policy_artifact, searcher, architectures, speed_bal, opp_simp_bal):
    encoder = policy_artifact["encoder"].to(searcher.device)
    policy = policy_artifact["policy"]
    features = []
    for arch in architectures:
        h_graph = searcher._get_embedding(encoder, arch)
        features.append(policy.lgbm_policy.extract_feature_vector(h_graph, {}))
    X = np.asarray(features, dtype=np.float32)
    comps = policy.predict_components(
        X,
        speed_bal=speed_bal,
        opp_simp_bal=opp_simp_bal,
    )
    return {
        "final": np.asarray(comps["final"], dtype=np.float32),
        "learnability": np.asarray(comps["learnability"], dtype=np.float32),
        "speed": np.asarray(comps["speed"], dtype=np.float32),
        "opp_simp_raw": np.asarray(comps["opp_simp_raw"], dtype=np.float32),
    }


def _safe_corr(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if len(a) < 2 or float(np.std(a)) < 1e-8 or float(np.std(b)) < 1e-8:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _error_metrics(predicted, actual):
    predicted = np.asarray(predicted, dtype=np.float32)
    actual = np.asarray(actual, dtype=np.float32)
    err = predicted - actual
    return {
        "mae": float(np.mean(np.abs(err))) if len(err) else 0.0,
        "rmse": float(np.sqrt(np.mean(err ** 2))) if len(err) else 0.0,
        "bias": float(np.mean(err)) if len(err) else 0.0,
        "max_abs": float(np.max(np.abs(err))) if len(err) else 0.0,
        "corr": _safe_corr(predicted, actual),
    }


def _run_policy_eval_sync(policy_artifact, n_tournaments, tournament_size, module_set, arch_size,
                          dataset_size, n_fights, max_duration, simp_opp_bal, client_id, on_progress):
    from search.rl_search import RLSearch

    eval_arena = _make_arena(
        n_fights=n_fights,
        dataset_size=dataset_size,
        architecture_size=arch_size,
        verbose=False,
        module_set=module_set,
        client_id=client_id,
    )
    searcher = RLSearch(arena=eval_arena, module_set=module_set, verbose=False)
    gen = Generator(generation_type="agnostic", module_types=module_set)
    speed_bal = float(eval_arena.speed_bal)
    rows = []
    all_pred = []
    all_actual = []
    all_pred_components = {"learnability": [], "speed": [], "opp_simp_raw": []}
    all_actual_components = {"learnability": [], "speed": [], "opp_simp_raw": []}

    on_progress({
        "type": "policy_eval_start",
        "n_tournaments": n_tournaments,
        "tournament_size": tournament_size,
        "total_fights": n_tournaments * (tournament_size * (tournament_size - 1) // 2),
        "speed_bal": speed_bal,
        "simp_opp_bal": simp_opp_bal,
        "normalization_group": normalization_key_for_modules(module_set),
    })

    for tournament_idx in range(1, n_tournaments + 1):
        architectures = []
        labels = []
        for arch_idx in range(tournament_size):
            architectures.append(gen.generate(arch_size))
            labels.append(f"T{tournament_idx} Arch {arch_idx + 1}")
            on_progress({
                "type": "policy_eval_generation",
                "tournament": tournament_idx,
                "current": arch_idx + 1,
                "total": tournament_size,
            })

        predictions = _policy_predict_for_architectures(
            policy_artifact,
            searcher,
            architectures,
            speed_bal=speed_bal,
            opp_simp_bal=simp_opp_bal,
        )
        on_progress({
            "type": "policy_eval_predictions",
            "tournament": tournament_idx,
            "predicted_scores": [float(v) for v in predictions["final"]],
        })

        def tournament_progress(message):
            if message.get("type") == "rl_tournament_fight":
                on_progress({
                    "type": "policy_eval_fight",
                    "tournament": tournament_idx,
                    "fight": message.get("fight"),
                    "total": message.get("total"),
                    "failed": message.get("failed", False),
                    "i": message.get("i"),
                    "j": message.get("j"),
                })

        tournament = searcher.run_full_tournament(
            architectures,
            labels=labels,
            progress_callback=tournament_progress,
            context={"tournament": tournament_idx},
            simp_opp_bal=simp_opp_bal,
            max_duration=max_duration,
        )

        tournament_rows = []
        for idx, label in enumerate(labels):
            actual = float(tournament["scores"][idx])
            predicted = float(predictions["final"][idx])
            row = {
                "tournament": tournament_idx,
                "arch": label,
                "nodes": len(architectures[idx].nodes),
                "predicted": predicted,
                "actual": actual,
                "delta": predicted - actual,
                "abs_delta": abs(predicted - actual),
                "predicted_learnability": float(predictions["learnability"][idx]),
                "actual_learnability": float(tournament["learnabilities"][idx]),
                "predicted_speed": float(predictions["speed"][idx]),
                "actual_speed": float(tournament["speeds"][idx]),
                "predicted_opp_simp_raw": float(predictions["opp_simp_raw"][idx]),
                "actual_opp_simp_raw": float(tournament.get("opp_simp_raw_bonuses", tournament["opp_simp_bonuses"])[idx]),
            }
            rows.append(row)
            tournament_rows.append(row)
            all_pred.append(predicted)
            all_actual.append(actual)
            all_pred_components["learnability"].append(row["predicted_learnability"])
            all_actual_components["learnability"].append(row["actual_learnability"])
            all_pred_components["speed"].append(row["predicted_speed"])
            all_actual_components["speed"].append(row["actual_speed"])
            all_pred_components["opp_simp_raw"].append(row["predicted_opp_simp_raw"])
            all_actual_components["opp_simp_raw"].append(row["actual_opp_simp_raw"])

        on_progress({
            "type": "policy_eval_tournament_done",
            "tournament": tournament_idx,
            "metrics": _error_metrics(
                [r["predicted"] for r in tournament_rows],
                [r["actual"] for r in tournament_rows],
            ),
            "rows": tournament_rows,
        })

    component_metrics = {
        key: _error_metrics(all_pred_components[key], all_actual_components[key])
        for key in all_pred_components
    }
    return {
        "metrics": _error_metrics(all_pred, all_actual),
        "component_metrics": component_metrics,
        "rows": rows,
        "n_samples": len(rows),
        "speed_bal": speed_bal,
        "simp_opp_bal": simp_opp_bal,
        "normalization_group": normalization_key_for_modules(module_set),
    }


@app.websocket("/ws/policy_eval")
async def policy_eval_ws(websocket: WebSocket):
    await websocket.accept()
    data = await websocket.receive_json()

    policy_id = data.get("policy_id")
    policy_artifact = policy_store.get(policy_id)
    if policy_artifact is None:
        await websocket.send_json({"type": "error", "message": "Selected Evolution Policy was not found on the backend."})
        await websocket.close()
        return

    n_tournaments = max(1, int(data.get("n_tournaments", 3)))
    tournament_size = max(2, int(data.get("tournament_size", 8)))
    module_set = data.get("module_set", "Unified")
    arch_size = max(3, int(data.get("arch_size", 12)))
    dataset_size = max(16, int(data.get("dataset_size", 320)))
    n_fights = max(1, int(data.get("n_fights", 1)))
    max_duration = max(1, int(data.get("max_duration", 600)))
    simp_opp_bal = float(data.get("simp_opp_bal", 0.2))
    client_id = data.get("client_id")

    import queue
    progress_queue = queue.Queue()

    def on_progress(message):
        progress_queue.put(message)

    loop = asyncio.get_event_loop()
    future = loop.run_in_executor(
        _search_executor,
        _run_policy_eval_sync,
        policy_artifact, n_tournaments, tournament_size, module_set, arch_size,
        dataset_size, n_fights, max_duration, simp_opp_bal, client_id, on_progress,
    )

    try:
        while not future.done():
            while True:
                try:
                    item = progress_queue.get_nowait()
                except queue.Empty:
                    break
                await websocket.send_json(item)
            await asyncio.sleep(0.1)

        result = await future
        while True:
            try:
                item = progress_queue.get_nowait()
            except queue.Empty:
                break
            await websocket.send_json(item)
        await websocket.send_json({"type": "policy_eval_done", "result": result})
    except Exception as exc:
        await websocket.send_json({"type": "error", "message": str(exc)})
    finally:
        await websocket.close()


@app.get("/api/generate")
def generate_architecture(
    module_set: str = Query("Unified", description="Module preset: Unified, Rich, or All"),
    arch_size: int = Query(12, ge=3, le=64, description="Target number of generated non-input nodes"),
):
    gen = Generator(generation_type="agnostic", module_types=module_set)
    arch = gen.generate(arch_size, randomize_n_nodes=False)

    arch_id = str(uuid.uuid4())
    arch_store[arch_id] = arch  

    return arch_to_graph_data(arch, arch_id=arch_id)


@app.get("/api/fight_viz")
async def fight_viz(arch_a_id: str = None, arch_b_id: str = None, module_set: str = Query("Unified", description="Module preset: Unified, Rich, or All")):
    gen = Generator(generation_type="agnostic", module_types=module_set)
    generating_A = True
    generating_B = True

    if arch_a_id and arch_a_id in arch_store:
        generating_A = False
        arch_a = arch_store[arch_a_id]

    if arch_b_id and arch_b_id in arch_store:
        generating_B = False
        arch_b = arch_store[arch_b_id]

    attempts = 0
    while attempts < 300:
        attempts += 1
        try:
            if generating_A:
                arch_a = gen.generate(12)
            if generating_B:
                arch_b = gen.generate(12)

            new_a_id = str(uuid.uuid4())
            new_b_id = str(uuid.uuid4())
            arch_store[new_a_id] = arch_a
            arch_store[new_b_id] = arch_b

            result = run_fight_visualization(
                arch_a, arch_b,
                max_iter=500, lr=5e-3, n_snapshots=50,
                generating_A=generating_A, generating_B=generating_B,
            )
            result["fight_a"]["arch_id"] = new_a_id
            result["fight_b"]["arch_id"] = new_b_id
            return result
        except Exception as e:
            print(f"Skipping bad architecture pair: {e}")
            continue

    raise HTTPException(status_code=500, detail="Could not produce a valid fight after 300 attempts")
    

@app.post("/api/mutate")
def mutate_arch(payload: dict):
    arch_id = payload.get("arch_id")
    action = payload.get("action")
    params = payload.get("params", {})

    if arch_id not in arch_store:
        raise HTTPException(status_code=404, detail="Architecture not found")

    arch = copy.deepcopy(arch_store[arch_id])
    mutator = Mutator(arch)

    try:
        if action == "add_node":
            module_type_str = params.get("module_type", "Activation")
            module_class = None
            from modules import ALL_MODULES as mod_list
            for mc in mod_list:
                if mc.__name__ == module_type_str:
                    module_class = mc
                    break
            if module_class is None:
                raise HTTPException(status_code=400, detail=f"Unknown module type: {module_type_str}")
            position_hint = params.get("position_hint")
            mutator.add_node(module_class, position_hint=position_hint)

        elif action == "remove_node":
            node_id = int(params["node_id"])
            mutator.remove_node(node_id)

        elif action == "replace_node":
            node_id = int(params["node_id"])
            module_type_str = params["module_type"]
            module_class = None
            from modules import ALL_MODULES as mod_list
            for mc in mod_list:
                if mc.__name__ == module_type_str:
                    module_class = mc
                    break
            if module_class is None:
                raise HTTPException(status_code=400, detail=f"Unknown module type: {module_type_str}")
            mutator.replace_node(node_id, module_class)

        elif action == "add_edge":
            source = int(params["source"])
            target = int(params["target"])
            mutator.add_edge(source, target)

        elif action == "remove_edge":
            source = int(params["source"])
            target = int(params["target"])
            mutator.remove_edge(source, target)

        elif action == "modify_params":
            node_id = int(params["node_id"])
            new_params = params.get("params", {})
            module = arch.nodes[node_id]['module']
            for key, value in new_params.items():
                _set_module_param(module, key, value)

        else:
            raise HTTPException(status_code=400, detail=f"Unknown action: {action}")

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    new_arch_id = str(uuid.uuid4())
    arch_store[new_arch_id] = arch
    data = arch_to_graph_data(arch, arch_id=new_arch_id)
    return data


@app.post("/api/crossover")
def crossover_archs(payload: dict):
    arch_a_id = payload.get("arch_a_id")
    arch_b_id = payload.get("arch_b_id")
    split_node = payload.get("split_node")

    if arch_a_id not in arch_store or arch_b_id not in arch_store:
        raise HTTPException(status_code=404, detail="Architecture not found")

    arch_a = copy.deepcopy(arch_store[arch_a_id])
    arch_b = copy.deepcopy(arch_store[arch_b_id])
    mutator = Mutator(arch_a)

    try:
        mutator.crossover(arch_b, split_node=split_node)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    new_arch_id = str(uuid.uuid4())
    arch_store[new_arch_id] = mutator.arch
    data = arch_to_graph_data(mutator.arch, arch_id=new_arch_id)
    data["mutations"] = mutator.record.describe()
    return data


@app.post("/api/mutate/batch")
def mutate_arch_batch(payload: dict):
    arch_id = payload.get("arch_id")
    mutations = payload.get("mutations", [])
    debug_prefix = "[mutate/batch]"

    if arch_id not in arch_store:
        raise HTTPException(status_code=404, detail="Architecture not found")

    arch = copy.deepcopy(arch_store[arch_id])
    mutator = Mutator(arch, validate=False)
    client_node_ids = {}

    def debug(message, **values):
        formatted_values = " ".join(f"{key}={value!r}" for key, value in values.items())
        print(f"{debug_prefix} {message} {formatted_values}".rstrip(), flush=True)

    def debug_arch_state(message):
        debug(
            message,
            nodes=[
                (node_id, arch.nodes[node_id]["module"].__class__.__name__)
                for node_id in arch.nodes
            ],
            edges=list(arch.edges),
            client_node_ids=dict(client_node_ids),
        )

    def find_module(name):
        from modules import ALL_MODULES as mod_list
        for mc in mod_list:
            if mc.__name__ == name:
                return mc
        return None

    def resolve_node_id(value):
        if isinstance(value, str) and value in client_node_ids:
            resolved = client_node_ids[value]
            debug("resolve client id", raw=value, resolved=resolved)
            return resolved
        try:
            resolved = int(value)
        except Exception as exc:
            debug("failed to resolve node id", raw=value, client_node_ids=dict(client_node_ids))
            raise ValueError(f"Cannot resolve node id {value!r}") from exc
        debug("resolve numeric id", raw=value, resolved=resolved)
        return resolved

    debug("start", arch_id=arch_id, mutation_count=len(mutations))
    debug_arch_state("initial state")

    for mutation_index, mutation in enumerate(mutations):
        action = mutation.get("action")
        params = mutation.get("params", {})
        debug("mutation", index=mutation_index, action=action, params=params)

        if action == "add_node":
            mc = find_module(params.get("module_type", "Activation"))
            if mc is None:
                raise HTTPException(status_code=400, detail=f"Unknown module type: {params.get('module_type')}")
            module_parameters = mc.random_parameters() or []
            module = mc(*module_parameters)
            new_id = arch.append_node(module)
            client_id = params.get("client_id")
            if client_id:
                client_node_ids[str(client_id)] = new_id
            mutator.record.add_mutation(f"Added {mc.__name__} node {new_id}")
            debug("added node", index=mutation_index, client_id=client_id, new_id=new_id, module=mc.__name__)

        elif action == "remove_node":
            node_id = params.get("node_id")
            if node_id is not None:
                mutator.remove_node(resolve_node_id(node_id))

        elif action == "replace_node":
            mc = find_module(params.get("module_type", ""))
            if mc is None:
                raise HTTPException(status_code=400, detail=f"Unknown module type: {params.get('module_type')}")
            node_id = params.get("node_id")
            if node_id is not None:
                mutator.replace_node(resolve_node_id(node_id), mc)

        elif action == "add_edge":
            source = params.get("source")
            target = params.get("target")
            if source is not None and target is not None:
                source_id = resolve_node_id(source)
                target_id = resolve_node_id(target)
                debug(
                    "resolved edge",
                    index=mutation_index,
                    raw_source=source,
                    raw_target=target,
                    source_id=source_id,
                    target_id=target_id,
                )
                if source_id == target_id:
                    debug_arch_state("self-edge rejected")
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            f"Cannot add self-edge on {_node_label_for_error(arch, source_id)} "
                            f"(raw source={source!r}, raw target={target!r}, mutation index={mutation_index})"
                        ),
                    )
                if source_id not in arch.nodes or target_id not in arch.nodes:
                    debug_arch_state("edge node not found")
                    raise HTTPException(status_code=400, detail=f"Cannot add edge {source_id} -> {target_id}: node not found")
                if arch.has_edge(source_id, target_id):
                    debug("edge already exists", source_id=source_id, target_id=target_id)
                    continue
                test_arch = copy.deepcopy(arch)
                test_arch.add_edge(source_id, target_id)
                if not nx.is_directed_acyclic_graph(test_arch):
                    debug_arch_state("cycle rejected")
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            f"Cannot add edge {_node_label_for_error(arch, source_id)} -> "
                            f"{_node_label_for_error(arch, target_id)} because it creates cycle: "
                            f"{_cycle_label_for_error(test_arch)}"
                        ),
                    )
                mutator.add_edge(source_id, target_id)
                debug("added edge", source_id=source_id, target_id=target_id, edges=list(arch.edges))

        elif action == "remove_edge":
            source = params.get("source")
            target = params.get("target")
            if source is not None and target is not None:
                mutator.remove_edge(resolve_node_id(source), resolve_node_id(target))

        elif action == "modify_params":
            node_id = int(params["node_id"])
            new_params = params.get("params", {})
            module = arch.nodes[node_id]['module']
            for key, value in new_params.items():
                _set_module_param(module, key, value)

    validation_errors = arch.validation_errors()
    if validation_errors:
        debug_arch_state("final validation failed")
        raise HTTPException(
            status_code=400,
            detail="Final architecture is invalid after mutations: " + "; ".join(validation_errors[:8]),
        )

    execution_errors = architecture_execution_errors(arch)
    if execution_errors:
        debug_arch_state("final execution validation failed")
        raise HTTPException(
            status_code=400,
            detail="Final architecture cannot execute after mutations: " + "; ".join(execution_errors[:4]),
        )

    new_arch_id = str(uuid.uuid4())
    arch_store[new_arch_id] = arch
    data = arch_to_graph_data(arch, arch_id=new_arch_id)
    data["mutations"] = mutator.record.describe()
    return data


@app.get("/api/arch_params/{arch_id}/{node_id}")
def get_node_params(arch_id: str, node_id: int):
    arch = arch_store.get(arch_id)
    if arch is None:
        raise HTTPException(status_code=404, detail="Architecture not found")

    if node_id not in arch.nodes:
        raise HTTPException(status_code=404, detail="Node not found")

    module = arch.nodes[node_id]['module']
    return {
        "node_id": node_id,
        "module_type": module.__class__.__name__,
        "mapping_type": module.mapping_type.name,
        "n_parameters": module.get_n_parameters(),
        "params": _module_params_for_ui(module),
    }


@app.websocket("/ws/tournament")
async def tournament_ws(websocket: WebSocket):
    print("[tournament] Waiting for client to connect...")
    await websocket.accept()
    print("[tournament] Client accepted, waiting for JSON config...")
    data = await websocket.receive_json()
    print(f"[tournament] Received config: n_random={data.get('n_random')}, module_set={data.get('module_set')}, "
          f"loaded={len(data.get('loaded_arch_ids', []))} archs, simp_opp_bal={data.get('simp_opp_bal')}")
    n_random        = data.get("n_random", 8)
    loaded_arch_ids = data.get("loaded_arch_ids", [])
    module_set      = data.get("module_set", "Unified")
    client_id       = data.get("client_id")
    simp_opp_bal    = data.get("simp_opp_bal", 0.2)
    max_duration    = data.get("max_duration", 600)
    arch_size       = data.get("arch_size", 12)

    tournament_arena = _make_arena(architecture_size=arch_size, verbose=False, module_set=module_set, client_id=client_id)
    gen = Generator(generation_type="agnostic", module_types=module_set)

    architectures = []
    arch_info     = []
    total_archs   = n_random + len(loaded_arch_ids)

    print(f"[tournament] Generating {n_random} random architectures...")
    for i in range(n_random):
        print(f"[tournament]   generating arch {i+1}/{n_random}...")
        try:
            arch = gen.generate(arch_size)
        except Exception as e:
            print(f"[tournament]   ERROR generating arch {i+1}: {e}")
            import traceback
            traceback.print_exc()
            await websocket.send_json({"type": "error", "message": f"Failed to generate architecture {i+1}: {e}"})
            await websocket.close()
            return
        arch_id = str(uuid.uuid4())
        arch_store[arch_id] = arch
        architectures.append(arch)
        arch_info.append({
            "id": len(arch_info), "arch_id": arch_id,
            "name": f"Random {i}", "source": "random",
        })
        print(f"[tournament]   arch {i+1} done, id={arch_id[:8]}")
        await websocket.send_json({
            "type": "generation_progress",
            "current": i + 1,
            "total": total_archs,
        })

    print(f"[tournament] Loading {len(loaded_arch_ids)} uploaded archs...")
    for entry in loaded_arch_ids:
        aid  = entry.get("arch_id", "")
        name = entry.get("name", aid[:8])
        arch = arch_store.get(aid)

        if arch is None:
            print(f"[tournament]   WARNING: arch_id {aid} not found in store, skipping")
            continue

        architectures.append(arch)
        arch_info.append({
            "id": len(arch_info), "arch_id": aid,
            "name": name, "source": "uploaded",
        })
        print(f"[tournament]   loaded {name}")

    n_archs = len(architectures)
    total   = n_archs * (n_archs - 1) // 2
    print(f"[tournament] Pool ready: {n_archs} archs, {total} fights")

    await websocket.send_json({
        "type": "init",
        "architectures": arch_info,
        "n_archs": n_archs,
        "total_fights": total,
        "normalization_group": normalization_key_for_modules(module_set),
    })
    print("[tournament] init message sent")

    if n_archs < 2:
        print("[tournament] Fewer than 2 archs, sending done")
        await websocket.send_json({"type": "done", "final_scores": []})
        return

    log_scores = [[0.0] * n_archs for _ in range(n_archs)]
    raw_learn_sum = [0.0] * n_archs
    raw_speed_sum = [0.0] * n_archs
    raw_time_sum  = [0.0] * n_archs
    fight_counts  = [0]   * n_archs

    def compute_scores(fc_arr, ls_mat, lrn_sum, spd_sum):
        if simp_opp_bal > 0:
            simps = []
            for col in range(n_archs):
                vals = [ls_mat[row][col] for row in range(n_archs) if row != col and ls_mat[row][col] != 0.0]
                if vals:
                    col_mean = sum(vals) / len(vals)
                else:
                    col_mean = tournament_arena.avg_simp
                simps.append((col_mean - tournament_arena.avg_simp) / tournament_arena.std_simp)
        s_arr, l_arr, sp_arr, t_arr = [], [], [], []
        for k in range(n_archs):
            fc  = max(fc_arr[k], 1)
            avg_s = spd_sum[k] / fc
            ns = (avg_s - tournament_arena.avg_speed) / tournament_arena.std_speed
            if simp_opp_bal > 0:
                total_w = 0.0
                w_count = 0
                for col in range(n_archs):
                    if col != k and ls_mat[k][col] != 0.0:
                        total_w += ls_mat[k][col] * math.exp(simps[col] * simp_opp_bal)
                        w_count += 1
                w_learn = total_w / max(w_count, 1)
                nl = (w_learn - tournament_arena.avg_learn) / tournament_arena.std_learn
            else:
                avg_l = lrn_sum[k] / fc
                nl = (avg_l - tournament_arena.avg_learn) / tournament_arena.std_learn
            comb = nl * (1.0 - tournament_arena.speed_bal) + ns * tournament_arena.speed_bal
            s_arr.append(comb)
            l_arr.append(nl)
            sp_arr.append(ns)
            t_arr.append(raw_time_sum[k] / fc)
        return s_arr, l_arr, sp_arr, t_arr

    fight = 0

    for i in range(n_archs):
        for j in range(i + 1, n_archs):
            fight += 1
            failed = False
            print(f"[tournament] Fight {fight}/{total}: arch {i} vs arch {j}")
            print(f"[tournament]   fitting arch {i} to learn {j} (max_duration={max_duration}s)...")

            _f_start = time.time()
            try:
                score_i, score_j, delay_i, delay_j = tournament_arena.get_scores(
                    architectures[i], architectures[j],
                    randomizeHP=True, pcp=0, get_delays=True, max_duration=max_duration,
                )
                _f_elapsed = time.time() - _f_start
                if _f_elapsed > max_duration * 0.9:
                    print(f"[tournament]   WARNING: fight {fight} lasted {_f_elapsed:.1f}s (limit {max_duration}s)")
                    failed = True
            except Exception as e:
                print(f"Fight {i} vs {j} errored: {e}, using fallback scores")
                import traceback
                traceback.print_exc()
                score_i, score_j = 1e-5, 1e-5
                delay_i, delay_j = 10.0, 10.0
                failed = True

            log_learn_i = math.log(max(score_i, 1e-10))
            log_learn_j = math.log(max(score_j, 1e-10))
            log_speed_i = math.log(max(1.0 / max(delay_i, 1e-6), 1e-10))
            log_speed_j = math.log(max(1.0 / max(delay_j, 1e-6), 1e-10))

            log_scores[i][j] = log_learn_i
            log_scores[j][i] = log_learn_j
            raw_learn_sum[i] += log_learn_i
            raw_learn_sum[j] += log_learn_j
            raw_speed_sum[i] += log_speed_i
            raw_speed_sum[j] += log_speed_j
            raw_time_sum[i]  += delay_i
            raw_time_sum[j]  += delay_j
            fight_counts[i]  += 1
            fight_counts[j]  += 1

            scores_arr, learns_arr, speeds_arr, times_arr = compute_scores(
                fight_counts, log_scores, raw_learn_sum, raw_speed_sum
            )

            fl_i = (log_learn_i - tournament_arena.avg_learn) / tournament_arena.std_learn
            fl_j = (log_learn_j - tournament_arena.avg_learn) / tournament_arena.std_learn
            fs_i = (log_speed_i - tournament_arena.avg_speed) / tournament_arena.std_speed
            fs_j = (log_speed_j - tournament_arena.avg_speed) / tournament_arena.std_speed
            fc_i = fl_i * (1.0 - tournament_arena.speed_bal) + fs_i * tournament_arena.speed_bal
            fc_j = fl_j * (1.0 - tournament_arena.speed_bal) + fs_j * tournament_arena.speed_bal

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

    final_scores, _, _, _ = compute_scores(
        fight_counts, log_scores, raw_learn_sum, raw_speed_sum
    )

    await websocket.send_json({
        "type": "done",
        "final_scores": [float(s) for s in final_scores],
        "architectures": arch_info,
    })


@app.get("/api/saved_archs")
def list_saved_archs():
    files = glob.glob("*.pkl")
    return {"files": files}

@app.get("/api/load_arch/{filename}")
def load_architecture(filename: str):
    if not os.path.exists(filename):
        return {"error": "File not found"}
        
    arch = Architecture.load(filename)
    return arch_to_graph_data(arch)

@app.post("/api/rl_search")
async def rl_search_endpoint(payload: dict):
    import time as time_module
    from search.rl_search import GNN_EMBEDDING_DIM
    n_phase_a = payload.get("n_phase_a_episodes", 50)
    n_phase_b = payload.get("n_phase_b_episodes", 50)
    n_candidates = payload.get("n_candidates_per_step", 10)
    module_set = payload.get("module_set", "Unified")
    client_id = payload.get("client_id")
    retrain_freq = payload.get("retrain_frequency", 50)

    search_arena = _make_arena(architecture_size=12, verbose=False, module_set=module_set, client_id=client_id)

    start_t = time_module.time()
    best_arch, history, lgbm_model = search_arena.rl_search(
        n_phase_a_episodes=n_phase_a,
        n_phase_b_episodes=n_phase_b,
        n_candidates_per_step=n_candidates,
        module_set=module_set,
        retrain_frequency=retrain_freq,
        verbose=False,
    )
    elapsed = time_module.time() - start_t

    arch_id = str(uuid.uuid4())
    arch_store[arch_id] = best_arch

    feat_imp = {}
    if lgbm_model is not None and lgbm_model.model is not None:
        feat_imp = lgbm_model.feature_importance(h_graph_dim=GNN_EMBEDDING_DIM)

    return {
        "best_architecture": arch_to_graph_data(best_arch, arch_id=arch_id),
        "training_history": history,
        "feature_importance": feat_imp,
        "elapsed_seconds": elapsed,
    }


import asyncio
from concurrent.futures import ThreadPoolExecutor

_search_executor = ThreadPoolExecutor(max_workers=1)


def _run_rl_search_sync(n_phase_a, n_phase_b, n_candidates, module_set, retrain_freq, arch_size, n_fights, dataset_size,
                        gnn_artifact=None, lgbm_policy=None, base_arch=None, skip_phase_b=False,
                        progress_callback=None, mode="legacy", tournament_size=8, simp_opp_bal=0.2,
                        n_gnn_epochs=50, policy_artifact=None, stop_event=None, acceptance_temperature=0.05,
                        train_on_tournament_archs=False, client_id=None):
    from search.rl_search import GNN_EMBEDDING_DIM, RLSearch

    search_arena = _make_arena(n_fights=n_fights, dataset_size=dataset_size, architecture_size=arch_size, verbose=False, module_set=module_set, client_id=client_id)
    searcher = RLSearch(arena=search_arena, module_set=module_set, verbose=False)

    if mode == "train_encoder":
        result = searcher.train_arch_encoder(
            n_epochs=n_phase_a,
            tournament_size=tournament_size,
            arch_size=arch_size,
            simp_opp_bal=simp_opp_bal,
            n_gnn_epochs=n_gnn_epochs,
            progress_callback=progress_callback,
            should_stop=stop_event.is_set if stop_event is not None else None,
        )
        result["lgbm_model"] = None
    elif mode == "evolve":
        if policy_artifact is not None:
            gnn_artifact = {
                "encoder": policy_artifact["encoder"],
                "predictor": policy_artifact.get("predictor"),
            }
        if gnn_artifact is None:
            raise ValueError("Evolve Architecture needs a trained GNN or Evolution Policy file.")
        result = searcher.evolve_architecture(
            encoder=gnn_artifact.get("encoder"),
            predictor=gnn_artifact.get("predictor"),
            n_epochs=n_phase_b,
            tournament_size=tournament_size,
            n_candidates_per_step=n_candidates,
            arch_size=arch_size,
            simp_opp_bal=simp_opp_bal,
            initial_arch=base_arch,
            initial_lgbm_policy=lgbm_policy,
            initial_evolution_policy=policy_artifact.get("policy") if policy_artifact else None,
            acceptance_temperature=acceptance_temperature,
            train_on_tournament_archs=train_on_tournament_archs,
            retrain_frequency=retrain_freq,
            progress_callback=progress_callback,
            should_stop=stop_event.is_set if stop_event is not None else None,
        )
    else:
        result = searcher.run(
            n_phase_a_episodes=n_phase_a,
            n_phase_b_episodes=n_phase_b,
            n_candidates_per_step=n_candidates,
            retrain_frequency=retrain_freq,
            n_gnn_epochs=n_gnn_epochs,
            train_frequency=max(10, n_phase_a // 10),
            initial_encoder=gnn_artifact.get("encoder") if gnn_artifact else None,
            initial_predictor=gnn_artifact.get("predictor") if gnn_artifact else None,
            initial_lgbm_policy=lgbm_policy,
            initial_arch=base_arch,
            skip_phase_b=skip_phase_b,
            progress_callback=progress_callback,
        )

    feat_imp = {}
    if result["lgbm_model"] is not None and result["lgbm_model"].model is not None:
        feat_imp = result["lgbm_model"].feature_importance(h_graph_dim=GNN_EMBEDDING_DIM)

    best_arch = result.get("best_architecture")
    return {
        "mode": mode,
        "best_arch": best_arch,
        "best_arch_json": searcher._arch_to_brief_json(best_arch) if best_arch is not None else None,
        "gnn_encoder": result["gnn_encoder"],
        "gnn_predictor": result.get("gnn_predictor"),
        "lgbm_model": result["lgbm_model"],
        "evolution_policy": result.get("evolution_policy"),
        "history": result["training_history"],
        "versions": result.get("versions", []),
        "top_architectures": result.get("top_architectures", []),
        "feature_importance": feat_imp,
        "final_reward": result["best_reward"],
        "interrupted": result.get("interrupted", False),
    }


@app.websocket("/ws/rl_search")
async def rl_search_ws(websocket: WebSocket):
    await websocket.accept()
    data = await websocket.receive_json()

    n_phase_a = data.get("n_phase_a_episodes", 50)
    n_phase_b = data.get("n_phase_b_episodes", 50)
    n_candidates = data.get("n_candidates_per_step", 10)
    module_set = data.get("module_set", "Unified")
    client_id = data.get("client_id")
    retrain_freq = data.get("retrain_frequency", 50)
    arch_size = data.get("arch_size", 12)
    n_fights = data.get("n_fights", 1)
    dataset_size = data.get("dataset_size", 320)
    mode = data.get("mode", "legacy")
    tournament_size = data.get("tournament_size", n_candidates)
    simp_opp_bal = data.get("simp_opp_bal", 0.2)
    n_gnn_epochs = data.get("n_gnn_epochs", 50)
    acceptance_temperature = float(data.get("acceptance_temperature", 0.05))
    train_on_tournament_archs = bool(data.get("train_on_tournament_archs", False))
    gnn_id = data.get("gnn_id")
    lgbm_id = data.get("lgbm_id")
    policy_id = data.get("policy_id")
    base_arch_id = data.get("base_arch_id")
    skip_phase_b = bool(data.get("skip_phase_b", False))

    gnn_artifact = None
    if gnn_id:
        gnn_artifact = gnn_store.get(gnn_id)
        if gnn_artifact is None:
            await websocket.send_json({"type": "error", "message": "Selected GNN was not found on the backend."})
            await websocket.close()
            return

    lgbm_policy = None
    if lgbm_id:
        lgbm_policy = lgbm_store.get(lgbm_id)
        if lgbm_policy is None:
            await websocket.send_json({"type": "error", "message": "Selected LGBM was not found on the backend."})
            await websocket.close()
            return

    policy_artifact = None
    if policy_id:
        policy_artifact = policy_store.get(policy_id)
        if policy_artifact is None:
            await websocket.send_json({"type": "error", "message": "Selected Evolution Policy was not found on the backend."})
            await websocket.close()
            return

    base_arch = None
    if base_arch_id:
        base_arch = arch_store.get(base_arch_id)
        if base_arch is None:
            await websocket.send_json({"type": "error", "message": "Selected base architecture was not found on the backend."})
            await websocket.close()
            return

    if mode == "train_encoder":
        start_message = f"Training architecture encoder: {n_phase_a} tournament epoch(s), {tournament_size} archs each."
    elif mode == "evolve":
        extra_learning = " Learning from all tournament architectures." if train_on_tournament_archs else ""
        start_message = f"Evolving architecture: initial tournament, then {n_phase_b} mutation epoch(s).{extra_learning}"
    else:
        start_message = f"Starting RL search: {'loaded GNN, skipping Phase A' if gnn_artifact else f'Phase A ({n_phase_a} eps)'} -> {'GNN download' if skip_phase_b else f'Phase B ({n_phase_b} eps)'}"
    await websocket.send_json({"type": "status", "message": start_message})

    loop = asyncio.get_event_loop()
    import queue
    import threading
    progress_queue = queue.Queue()
    stop_event = threading.Event()

    def on_progress(msg):
        progress_queue.put(msg)

    async def listen_for_stop():
        try:
            while not stop_event.is_set():
                msg = await websocket.receive_json()
                if msg.get("type") == "stop":
                    stop_event.set()
                    try:
                        await websocket.send_json({"type": "status", "message": "Stop requested. Finishing the current step and saving partial results..."})
                    except Exception:
                        pass
                    return
        except Exception:
            stop_event.set()

    future = loop.run_in_executor(
        _search_executor,
        _run_rl_search_sync,
        n_phase_a, n_phase_b, n_candidates, module_set, retrain_freq, arch_size, n_fights, dataset_size,
        gnn_artifact, lgbm_policy, base_arch, skip_phase_b,
        on_progress,
        mode, tournament_size, simp_opp_bal, n_gnn_epochs,
        policy_artifact, stop_event, acceptance_temperature, train_on_tournament_archs, client_id,
    )
    stop_task = asyncio.create_task(listen_for_stop())
    top_architecture_ids = {}

    def serialize_top_architectures(rows):
        payload = []
        for row in rows or []:
            arch = row.get("architecture")
            signature = row.get("signature")
            arch_id = top_architecture_ids.get(signature)
            if arch_id is None and arch is not None:
                arch_id = str(uuid.uuid4())
                arch_store[arch_id] = arch
                top_architecture_ids[signature] = arch_id
            payload.append({
                "rank": row.get("rank"),
                "role": row.get("role"),
                "source": row.get("source"),
                "epoch": row.get("epoch"),
                "eval_role": row.get("eval_role"),
                "score": row.get("score"),
                "learnability": row.get("learnability"),
                "speed": row.get("speed"),
                "opp_simp_raw": row.get("opp_simp_raw"),
                "n_nodes": row.get("n_nodes"),
                "n_params": row.get("n_params"),
                "arch_id": arch_id,
                "graph": row.get("arch_json"),
            })
        return payload

    def prepare_progress_item(item):
        if isinstance(item, dict) and item.get("type") == "evolve_top_architectures":
            return {
                **item,
                "top_architectures": serialize_top_architectures(item.get("top_architectures")),
            }
        return item

    try:
        while not future.done():
            while True:
                try:
                    item = progress_queue.get_nowait()
                except queue.Empty:
                    break
                try:
                    await websocket.send_json(prepare_progress_item(item))
                except Exception:
                    stop_event.set()
                    future.cancel()
                    stop_task.cancel()
                    return
            await asyncio.sleep(0.1)

        result = await future

        while True:
            try:
                item = progress_queue.get_nowait()
            except queue.Empty:
                break
            try:
                await websocket.send_json(prepare_progress_item(item))
            except Exception:
                stop_task.cancel()
                return
        stop_task.cancel()
    except Exception as exc:
        stop_task.cancel()
        try:
            await websocket.send_json({
                "type": "error",
                "message": str(exc),
            })
        except Exception:
            pass
        try:
            await websocket.close()
        except Exception:
            pass
        return

    arch_id = None
    if result.get("best_arch") is not None:
        arch_id = str(uuid.uuid4())
        arch_store[arch_id] = result["best_arch"]
    saved_gnn_id = str(uuid.uuid4())
    gnn_store[saved_gnn_id] = {
        "encoder": result["gnn_encoder"],
        "predictor": result.get("gnn_predictor"),
        "metadata": {"source": "rl_search", "arch_id": arch_id},
    }
    saved_lgbm_id = None
    if result.get("lgbm_model") is not None:
        saved_lgbm_id = str(uuid.uuid4())
        lgbm_store[saved_lgbm_id] = result["lgbm_model"]
    saved_policy_id = None
    if result.get("evolution_policy") is not None:
        saved_policy_id = str(uuid.uuid4())
        policy_store[saved_policy_id] = {
            "encoder": result["gnn_encoder"],
            "predictor": result.get("gnn_predictor"),
            "policy": result["evolution_policy"],
            "metadata": {"source": "rl_search", "arch_id": arch_id, "interrupted": result.get("interrupted", False)},
        }

    feat_imp = result.get("feature_importance", {})
    versions_payload = []
    for version in result.get("versions", []):
        version_arch = version.get("architecture")
        if version_arch is None:
            continue
        version_arch_id = str(uuid.uuid4())
        arch_store[version_arch_id] = version_arch
        versions_payload.append({
            "version": version.get("version"),
            "label": version.get("label"),
            "score": version.get("score"),
            "arena_delta": version.get("arena_delta"),
            "accepted": version.get("accepted", True),
            "exploratory": version.get("exploratory", False),
            "acceptance_probability": version.get("acceptance_probability"),
            "true_learnability": version.get("true_learnability"),
            "true_speed": version.get("true_speed"),
            "true_opp_simp_bonus": version.get("true_opp_simp_bonus"),
            "true_opp_simp_raw_bonus": version.get("true_opp_simp_raw_bonus"),
            "prediction_error": version.get("prediction_error"),
            "arch_id": version_arch_id,
            "graph": version.get("arch_json"),
        })
    top_architectures_payload = serialize_top_architectures(result.get("top_architectures", []))
    await websocket.send_json({
        "type": "done",
        "arch_id": arch_id,
        "gnn_id": saved_gnn_id,
        "lgbm_id": saved_lgbm_id,
        "policy_id": saved_policy_id,
        "interrupted": result.get("interrupted", False),
        "best_reward": result["final_reward"],
        "feature_importance": feat_imp,
        "history": result["history"],
        "versions": versions_payload,
        "top_architectures": top_architectures_payload,
        "best_arch_json": result.get("best_arch_json"),
    })


@app.get("/")
def root():
    return {
        "status": "Emernet API running",
        "endpoints": ["/api/generate", "/api/mutate", "/api/crossover",
                       "/api/arch_params", "/ws/tournament", "/ws/rl_search",
                       "/api/rl_search", "/docs"],
    }
