import copy
import random
import time
import torch
import networkx
from graph.architecture import Architecture
from graph.executor import Executor
from modules.input import Input
from modules.base import MappingType
from modules import ALL_MODULES


class ArchValidator:
    def __init__(self, arch):
        self.arch = arch

    def can_add_node(self, module_type):
        return True

    def can_remove_node(self, node_id):
        # Only forbid removing the mandatory input node (id=0)
        if node_id == 0:
            return False
        return True

    def can_add_edge(self, source, target):
        if source == target:
            return False
        if source not in self.arch.nodes or target not in self.arch.nodes:
            return False
        if self.arch.nodes[target]['module'].mapping_type == MappingType.SOURCE:
            return False
        if self.arch.has_edge(source, target):
            return False
        test = copy.deepcopy(self.arch)
        test.add_edge(source, target)
        return networkx.is_directed_acyclic_graph(test) and test.isValid(verbose=False)

    def can_remove_edge(self, source, target):
        if not self.arch.has_edge(source, target):
            return False
        test = copy.deepcopy(self.arch)
        test.remove_edge(source, target)
        return test.isValid(verbose=False)


class MutationRecord:
    def __init__(self, original_arch):
        self.original = copy.deepcopy(original_arch)
        self.mutations = []
        self.timestamps = []

    def add_mutation(self, description):
        self.mutations.append(description)
        self.timestamps.append(time.time())

    def undo(self):
        return copy.deepcopy(self.original)

    def describe(self):
        return "\n".join(self.mutations)


class Mutator:
    def __init__(self, arch, validate=True):
        self.arch = arch
        self.validator = ArchValidator(arch)
        self.record = MutationRecord(arch)
        self.validate = validate

    def _find_insertion_point(self):
        topo = list(networkx.topological_sort(self.arch))
        mid_idx = len(topo) // 2
        return topo[mid_idx]

    def add_node(self, module_type, position_hint=None):
        backup = copy.deepcopy(self.arch)
        module_parameters = module_type.random_parameters() or []
        module = module_type(*module_parameters) if callable(module_type) else module_type

        new_id = self.arch.append_node(module)

        edges = list(self.arch.edges)
        if edges:
            pred, succ = random.choice(edges)
            self.arch.remove_edge(pred, succ)
            if module.mapping_type == MappingType.SOURCE:
                self.arch.add_edge(new_id, succ)
                if self.validate and not self.arch.isValid(verbose=False):
                    self.arch = backup
                    self.validator = ArchValidator(self.arch)
                    raise Exception("Mutation produced invalid architecture")
            else:
                self.arch.add_edge(pred, new_id)
                self.arch.add_edge(new_id, succ)
        else:
            if module.mapping_type != MappingType.SOURCE:
                self.arch.add_edge(0, new_id)

        if self.validate and not self.arch.isValid(verbose=False):
            self.arch = backup
            self.validator = ArchValidator(self.arch)
            raise Exception("Mutation produced invalid architecture")

        self.record.add_mutation(f"Added {module_type.__name__} node {new_id}")
        return new_id

    def remove_node(self, node_id):
        if not self.validator.can_remove_node(node_id):
            raise Exception(f"Cannot remove node {node_id}")

        predecessors = list(self.arch.predecessors(node_id))
        successors = list(self.arch.successors(node_id))

        self.arch.remove_node(node_id)

        for pred in predecessors:
            if pred in self.arch.nodes:
                for succ in successors:
                    if succ in self.arch.nodes and pred != succ:
                        try:
                            if not self.arch.has_edge(pred, succ):
                                self.arch.add_edge(pred, succ)
                        except Exception:
                            pass

        if self.validate and not self.arch.isValid(verbose=False):
            raise Exception("Mutation produced invalid architecture")

        self.record.add_mutation(f"Removed node {node_id}")
        return self.arch

    def remove_node(self, node_id):
        backup = copy.deepcopy(self.arch)
        if not self.validator.can_remove_node(node_id):
            raise Exception(f"Cannot remove node {node_id}")

        predecessors = list(self.arch.predecessors(node_id))
        successors = list(self.arch.successors(node_id))

        self.arch.remove_node(node_id)

        for pred in predecessors:
            if pred in self.arch.nodes:
                for succ in successors:
                    if succ in self.arch.nodes and pred != succ:
                        try:
                            if not self.arch.has_edge(pred, succ):
                                self.arch.add_edge(pred, succ)
                        except Exception:
                            pass

        if self.validate and not self.arch.isValid(verbose=False):
            raise Exception("Mutation produced invalid architecture")

        self.record.add_mutation(f"Removed node {node_id}")
        return self.arch

    def replace_node(self, node_id, new_module_type):
        old_module = self.arch.nodes[node_id]['module']
        new_mapping = getattr(new_module_type, "_mapping_type", None)
        if old_module.mapping_type != MappingType.SOURCE and new_mapping == MappingType.SOURCE:
            raise Exception("Cannot replace a non-source node with a source node")
        backup_module = copy.deepcopy(self.arch.nodes[node_id]['module'])
        module_parameters = new_module_type.random_parameters() or []
        module = new_module_type(*module_parameters) if callable(new_module_type) else new_module_type

        self.arch.nodes[node_id]['module'] = module

        if self.validate and not self.arch.isValid(verbose=False):
            self.arch.nodes[node_id]['module'] = backup_module
            raise Exception("Mutation produced invalid architecture")

        self.record.add_mutation(f"Replaced node {node_id} with {new_module_type.__name__}")
        return self.arch

    def add_edge(self, source, target):
        if self.validate and not self.validator.can_add_edge(source, target):
            raise Exception(f"Cannot add edge {source} -> {target}")
        self.arch.add_edge(source, target)
        self.record.add_mutation(f"Added edge {source} -> {target}")
        return self.arch

    def remove_edge(self, source, target):
        if self.validate and not self.validator.can_remove_edge(source, target):
            raise Exception(f"Cannot remove edge {source} -> {target}")
        self.arch.remove_edge(source, target)
        self.record.add_mutation(f"Removed edge {source} -> {target}")
        return self.arch

    def mutate_parameters(self, node_id, mutation_scale=0.1):
        module = self.arch.nodes[node_id]['module']
        for param in module.parameters(recurse=False):
            if param.requires_grad:
                with torch.no_grad():
                    param.add_(torch.randn_like(param) * mutation_scale)
        self.record.add_mutation(f"Mutated parameters of node {node_id}")

    def swap_modules(self, node_a, node_b):
        module_a = self.arch.nodes[node_a]['module']
        module_b = self.arch.nodes[node_b]['module']
        self.arch.nodes[node_a]['module'] = module_b
        self.arch.nodes[node_b]['module'] = module_a
        if not self.arch.isValid():
            self.arch.nodes[node_a]['module'] = module_a
            self.arch.nodes[node_b]['module'] = module_b
            raise Exception("Swap produced invalid architecture")
        self.record.add_mutation(f"Swapped nodes {node_a} and {node_b}")
        return self.arch

    def crossover(self, other_arch, split_node=None):
        if split_node is None:
            split_node = min(self.arch.nodes, key=lambda n: abs(n - len(self.arch.nodes) // 2))

        self_topo = list(networkx.topological_sort(self.arch))
        other_topo = list(networkx.topological_sort(other_arch))

        split_idx = self_topo.index(split_node) if split_node in self_topo else len(self_topo) // 2

        prefix_ids = self_topo[:split_idx]
        suffix_ids = other_topo[split_idx:]

        combined = Architecture()
        id_map = {}

        for node_id in prefix_ids:
            module = copy.deepcopy(self.arch.nodes[node_id]['module'])
            new_id = combined.append_node(module)
            id_map[node_id] = new_id

        for node_id in suffix_ids:
            if node_id == 0:
                continue
            module = copy.deepcopy(other_arch.nodes[node_id]['module'])
            new_id = combined.append_node(module)
            id_map[node_id] = new_id

        for u, v in self.arch.edges:
            if u in id_map and v in id_map:
                combined.add_edge(id_map[u], id_map[v])

        for u, v in other_arch.edges:
            if u in id_map and v in id_map:
                if not combined.has_edge(id_map[u], id_map[v]) and u != v:
                    try:
                        combined.add_edge(id_map[u], id_map[v])
                    except Exception:
                        pass

        if not combined.isValid():
            raise Exception("Crossover produced invalid architecture")

        self.arch = combined
        self.record.add_mutation(f"Crossover at split_node {split_node}")
        return self.arch
