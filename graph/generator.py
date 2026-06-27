import random
import networkx
import torch
from graph.architecture import Architecture
from graph.executor import Executor
from modules.input import Input
from modules.base import MappingType
from modules import ALL_MODULES, UNIFIED_PRESET, RICH_PRESET


class Generator:
    def __init__(self, generation_type="agnostic", module_types="Unified"):
        self.generation_type = generation_type
        if module_types == "Unified":
            self.available_modules = UNIFIED_PRESET
        elif module_types == "Rich":
            self.available_modules = RICH_PRESET
        else:
            self.available_modules = ALL_MODULES

    def _weighted_choice(self):
        weights = [getattr(m, 'selection_weight', 1.0) for m in self.available_modules]
        return random.choices(self.available_modules, weights=weights, k=1)[0]

    def _is_nonlinear(self, executor, shape=(15, 18), min_threshold=1e-5, max_threshold=1e9):
        dummy_input = torch.randn(1, *shape)
        with torch.no_grad():
            executor.forward(dummy_input)

        executor.randomize_weights()

        x1 = torch.randn(1, *shape)
        x2 = torch.randn(1, *shape)
        x_mid = (x1 + x2) / 2.0

        x_batch = torch.cat([x1, x2, x_mid], dim=0)

        with torch.no_grad():
            out = executor.forward(x_batch)

        out_tensor = out[0]

        expected_linear_mid = (out_tensor[0] + out_tensor[1]) / 2.0
        actual_mid = out_tensor[2]

        deviation = torch.abs(actual_mid - expected_linear_mid).mean().item()

        return deviation > min_threshold and deviation < max_threshold

    def generate(self, n_nodes=12, randomize_n_nodes=True) -> Architecture:
        generated = False
        iters = 0
        if randomize_n_nodes:
            perturbations = [-3, -2, -1, 0, 1, 2, 3, 4]
            n_nodes += random.choice(perturbations)
        if self.generation_type == "dense":
            while not generated and iters < 50:
                iters += 1
                try:
                    arch = self.generate_dense(n_nodes)
                    if len(arch.nodes) < 3 or arch.parameter_count() < 16:
                        continue
                    test_input = torch.randn(2, 15, 18)
                    ex = Executor(arch)
                    if not self._is_nonlinear(ex):
                        continue

                    test_input = torch.randn(2, 15, 18)
                    out = ex.forward(test_input)
                    if torch.isfinite(out[0]).all():
                        return arch
                except Exception as e:
                    pass
        elif self.generation_type == "agnostic":
            while not generated and iters < 50:
                iters += 1
                try:
                    arch = self.generate_order_agnostic(n_nodes)
                    if len(arch.nodes) < 3 or arch.parameter_count() < 16:
                        continue
                    test_input = torch.randn(2, 15, 18)
                    ex = Executor(arch)
                    if not self._is_nonlinear(ex):
                        continue
                    out = ex.forward(test_input)
                    if torch.isfinite(out[0]).all():
                        return arch
                except Exception as e:
                    pass
        else:
            raise Exception(f"Unknown generation type {self.generation_type}")

        raise Exception(f"Could not generate a valid architecture in {iters} attempts")


    def generate_dense(self, n_nodes=16) -> Architecture:
        self.architecture = Architecture()
        self.architecture.add_node(0, Input())
        stopflag = False
        rooting_p = 1
        while not stopflag:
            module_type = self._weighted_choice()
            module_parameters = module_type.random_parameters() or []
            lastnode_id = self.architecture.append_node(module_type(*module_parameters))

            linked = False
            if module_type._mapping_type != MappingType.SOURCE:
                for node in list(self.architecture.nodes)[:-1]:
                    if random.random() < rooting_p:
                        self.architecture.add_edge(node, lastnode_id)
                        rooting_p = rooting_p * 0.96
                        linked = True
                if not linked:
                    stopflag = True

        outputnode = 0
        outputancestors = 0
        for node in self.architecture.nodes:
            ancestors = networkx.ancestors(self.architecture, node)
            if len(ancestors) > outputancestors and 0 in ancestors:
                outputnode = node
                outputancestors = len(ancestors)

        nodesToRemove = []
        for node in self.architecture.nodes:
            if node not in networkx.ancestors(self.architecture, outputnode) and node != outputnode:
                nodesToRemove.append(node)
        for node in nodesToRemove:
            self.architecture.remove_node(node)
        if not self.architecture.isValid(verbose=False):
            raise Exception("The architecture is not valid after the generation")
        return self.architecture

    def generate_order_agnostic(self, n_nodes=16) -> Architecture:
        self.architecture = Architecture()
        self.architecture.add_node(0, Input())

        rooting_p = random.choice([0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3, 0.35, 0.4, 0.45])
        for i in range(n_nodes):
            module_type = self._weighted_choice()
            module_parameters = module_type.random_parameters() or []
            lastnode_id = self.architecture.append_node(module_type(*module_parameters))

        for source_node in list(self.architecture.nodes):
            for target_node in list(self.architecture.nodes):
                if target_node != source_node and self.architecture.nodes[target_node]['module'].mapping_type != MappingType.SOURCE:
                    if random.random() < rooting_p and target_node not in list(networkx.ancestors(self.architecture, source_node)):
                        self.architecture.add_edge(source_node, target_node)

        for node in list(self.architecture.nodes):
            if self.architecture.nodes[node]['module'].mapping_type == MappingType.SOURCE:
                continue
            ancestors = networkx.ancestors(self.architecture, node)
            has_source = any(
                self.architecture.nodes[a]['module'].mapping_type == MappingType.SOURCE
                for a in ancestors
            )
            if has_source:
                continue

            candidates = [
                n for n in self.architecture.nodes
                if n != node
                and n not in ancestors
                and node not in networkx.ancestors(self.architecture, n)
            ]
            random.shuffle(candidates)
            for c in candidates:
                if random.random() < rooting_p:
                    self.architecture.add_edge(c, node)
                    break
            else:
                if candidates:
                    self.architecture.add_edge(candidates[0], node)

        outputnode = 0
        outputancestors = 0
        for node in self.architecture.nodes:
            ancestors = networkx.ancestors(self.architecture, node)
            if len(ancestors) > outputancestors and 0 in ancestors and self.architecture.nodes[node]['module'].mapping_type != MappingType.SOURCE:
                outputnode = node
                outputancestors = len(ancestors)

        nodesToRemove = []
        for node in self.architecture.nodes:
            if node not in networkx.ancestors(self.architecture, outputnode) and node != outputnode:
                nodesToRemove.append(node)
        for node in nodesToRemove:
            self.architecture.remove_node(node)
        if not self.architecture.isValid(verbose=False):
            raise Exception("The architecture is not valid after the generation")
        return self.architecture
