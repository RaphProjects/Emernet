import torch
import torch.nn as nn
import networkx
import copy
import math

from graph.architecture import Architecture
from modules.base import ModuleType, MappingType


MODULE_TYPE_NAMES = [t for t in ModuleType]
MAPPING_TYPE_NAMES = [t for t in MappingType]
NODE_FEATURE_DIM = len(MODULE_TYPE_NAMES) + len(MAPPING_TYPE_NAMES) + 1


def build_node_features(arch: Architecture, node_order=None) -> torch.Tensor:
    n_nodes = len(arch.nodes)
    if n_nodes == 0:
        return torch.zeros((0, NODE_FEATURE_DIM))

    features = []
    ordered_nodes = node_order if node_order is not None else list(arch.nodes)
    for node_id in ordered_nodes:
        attr = arch.nodes[node_id]
        if "module" not in attr:
            continue

        mod = attr["module"]

        mt_one_hot = torch.zeros(len(MODULE_TYPE_NAMES))
        for i, mt in enumerate(MODULE_TYPE_NAMES):
            if mod.module_type == mt:
                mt_one_hot[i] = 1.0
                break

        mp_one_hot = torch.zeros(len(MAPPING_TYPE_NAMES))
        for i, mp in enumerate(MAPPING_TYPE_NAMES):
            if mod.mapping_type == mp:
                mp_one_hot[i] = 1.0
                break

        n_params = torch.tensor([float(mod.get_n_parameters())])

        f = torch.cat([mt_one_hot, mp_one_hot, n_params])
        features.append(f)

    if not features:
        return torch.zeros((1, NODE_FEATURE_DIM))

    return torch.stack(features)


def topological_order(arch: Architecture):
    return list(networkx.topological_sort(arch))


def predecessors_in_order(arch: Architecture, topo_order: list[int]):
    pred_map = {}
    for node_id in topo_order:
        pred_map[node_id] = [p for p in arch.predecessors(node_id) if p in topo_order]
    return pred_map


class ArchEncoder(nn.Module):
    def __init__(self, node_feat_dim, hidden_dim=128, n_layers=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.input_proj = nn.Linear(node_feat_dim, hidden_dim)

        self.self_weights = nn.ModuleList()
        self.neigh_weights = nn.ModuleList()
        for _ in range(n_layers):
            self.self_weights.append(nn.Linear(hidden_dim, hidden_dim, bias=False))
            self.neigh_weights.append(nn.Linear(hidden_dim, hidden_dim, bias=False))

    def forward(self, node_features: torch.Tensor, topo_order: list[int],
                pred_map: dict[int, list[int]]) -> torch.Tensor:
        h = torch.relu(self.input_proj(node_features))

        for layer_idx in range(self.n_layers):
            h_new = torch.zeros_like(h)
            for pos, node_id in enumerate(topo_order):
                h_self = self.self_weights[layer_idx](h[pos])

                preds = pred_map[node_id]
                if preds:
                    pred_indices = [topo_order.index(p) for p in preds]
                    h_neigh = torch.mean(h[pred_indices], dim=0)
                else:
                    h_neigh = torch.zeros_like(h_self)
                h_neigh_transformed = self.neigh_weights[layer_idx](h_neigh)

                h_new[pos] = torch.relu(h_self + h_neigh_transformed)
            h = h_new

        return h

    def graph_readout(self, h: torch.Tensor) -> torch.Tensor:
        h_mean = torch.mean(h, dim=0)
        h_max = torch.max(h, dim=0)[0]
        h_sum = torch.sum(h, dim=0)
        return torch.cat([h_mean, h_max, h_sum])


class RewardPredictor(nn.Module):
    def __init__(self, input_dim=384):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 96),
            nn.ReLU(),
            nn.Linear(96, 48),
            nn.ReLU(),
            nn.Linear(48, 1),
        )

    def forward(self, h_graph: torch.Tensor) -> torch.Tensor:
        return self.net(h_graph).squeeze(-1)


def encode_architecture(arch: Architecture, encoder: ArchEncoder,
                        device: torch.device) -> torch.Tensor:
    topo = topological_order(arch)
    features = build_node_features(arch, topo).to(device)
    pred_map = predecessors_in_order(arch, topo)
    h = encoder(features, topo, pred_map)
    h_graph = encoder.graph_readout(h)
    return h_graph


def batch_encode(archs: list[Architecture], encoder: ArchEncoder,
                 device: torch.device) -> torch.Tensor:
    h_graphs = []
    for arch in archs:
        h_graphs.append(encode_architecture(arch, encoder, device))
    return torch.stack(h_graphs)


def train_gnn(encoder: ArchEncoder, predictor: RewardPredictor,
              archs: list[Architecture], rewards: list[float],
              n_epochs=100, lr=0.001, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    encoder.to(device)
    predictor.to(device)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=lr
    )

    features_list = []
    topo_list = []
    pred_map_list = []

    for arch in archs:
        topo = topological_order(arch)
        f = build_node_features(arch, topo)
        pred_map = predecessors_in_order(arch, topo)
        features_list.append(f.to(device))
        topo_list.append(topo)
        pred_map_list.append(pred_map)

    targets = torch.tensor(rewards, dtype=torch.float32, device=device)

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        h_graphs = []

        for i in range(len(archs)):
            h = encoder(features_list[i], topo_list[i], pred_map_list[i])
            h_graph = encoder.graph_readout(h)
            h_graphs.append(h_graph)

        h_graphs = torch.stack(h_graphs)
        preds = predictor(h_graphs)
        loss = nn.functional.mse_loss(preds, targets)

        loss.backward()
        optimizer.step()

    return loss.item()
