import random
import copy
import math
import torch
import networkx

from graph.architecture import Architecture
from graph.mutator import Mutator
from graph.validation import architecture_execution_errors
from modules.base import MappingType
from modules import UNIFIED_PRESET


def _arch_signature(arch: Architecture):
    node_sig = tuple(
        (n, type(arch.nodes[n]["module"]).__name__)
        for n in sorted(arch.nodes)
    )
    edge_sig = tuple(sorted((int(s), int(t)) for s, t in arch.edges))
    return node_sig, edge_sig


def propose_random_mutations(arch: Architecture, n_candidates=10) -> list[dict]:
    candidates = []
    original_signature = _arch_signature(arch)

    def add_candidate(candidate):
        mutated_arch = candidate["mutated_arch"]
        if _arch_signature(mutated_arch) == original_signature:
            return
        execution_errors = architecture_execution_errors(mutated_arch)
        if execution_errors:
            candidate["is_valid"] = False
            candidate["execution_errors"] = execution_errors
            return
        candidates.append(candidate)

    def apply_one_step(mutator):
        working_arch = mutator.arch
        node_ids = list(working_arch.nodes)
        output_id = working_arch.get_Output_id()
        op = random.choices(
            ["add_node", "remove_node", "replace_node", "add_edge", "remove_edge"],
            weights=[0.25, 0.15, 0.25, 0.2, 0.15],
            k=1
        )[0]

        if op == "add_node":
            mt = random.choice(UNIFIED_PRESET)
            mutator.add_node(mt)
            return {
                "mutation_type": "add_node",
                "target_module_type": mt.__name__ if hasattr(mt, '__name__') else str(mt),
            }

        if op == "remove_node":
            valid_targets = [n for n in node_ids if n != 0 and n != output_id]
            if not valid_targets:
                return None
            target = random.choice(valid_targets)
            removed_module_type = type(working_arch.nodes[target]['module']).__name__
            mutator.remove_node(target)
            return {
                "mutation_type": "remove_node",
                "target_module_type": removed_module_type,
            }

        if op == "replace_node":
            valid_targets = [n for n in node_ids if n != 0]
            if not valid_targets:
                return None
            target = random.choice(valid_targets)
            old_mapping = working_arch.nodes[target]['module'].mapping_type
            module_pool = [
                mt for mt in UNIFIED_PRESET
                if not (old_mapping != MappingType.SOURCE and getattr(mt, "_mapping_type", None) == MappingType.SOURCE)
            ]
            if not module_pool:
                return None
            new_mt = random.choice(module_pool)
            mutator.replace_node(target, new_mt)
            return {
                "mutation_type": "replace_node",
                "target_module_type": new_mt.__name__ if hasattr(new_mt, '__name__') else str(new_mt),
            }

        if op == "add_edge":
            sources = [n for n in node_ids]
            targets = [
                n for n in node_ids
                if n != 0 and working_arch.nodes[n]['module'].mapping_type != MappingType.SOURCE
            ]
            if not sources or not targets:
                return None
            random.shuffle(sources)
            random.shuffle(targets)
            for s in sources:
                for t in targets:
                    if mutator.validator.can_add_edge(s, t):
                        mutator.add_edge(s, t)
                        return {"mutation_type": "add_edge", "target_module_type": ""}
            return None

        if op == "remove_edge":
            edges = list(working_arch.edges)
            valid_edges = [
                (s, t) for s, t in edges
                if s != 0 and mutator.validator.can_remove_edge(s, t)
            ]
            if not valid_edges:
                return None
            s, t = random.choice(valid_edges)
            mutator.remove_edge(s, t)
            return {"mutation_type": "remove_edge", "target_module_type": ""}

        return None

    for _ in range(n_candidates * 20):
        if len(candidates) >= n_candidates:
            break

        mutator = Mutator(copy.deepcopy(arch))
        n_steps = random.choices([1, 2, 3], weights=[0.6, 0.3, 0.1], k=1)[0]
        steps = []

        try:
            for _step in range(n_steps):
                before = _arch_signature(mutator.arch)
                step = apply_one_step(mutator)
                if step is None:
                    continue
                after = _arch_signature(mutator.arch)
                if after != before:
                    steps.append(step)

            if not steps:
                continue

            mutation_sequence = [s["mutation_type"] for s in steps]
            target_sequence = [s.get("target_module_type", "") for s in steps]
            primary_step = steps[-1]
            add_candidate({
                "mutation_type": "compound" if len(steps) > 1 else primary_step["mutation_type"],
                "target_module_type": primary_step.get("target_module_type", ""),
                "mutation_sequence": mutation_sequence,
                "target_module_sequence": target_sequence,
                "n_mutation_steps": len(steps),
                "n_nodes_after": len(mutator.arch.nodes),
                "n_params_after": mutator.arch.parameter_count(),
                "is_valid": True,
                "mutated_arch": copy.deepcopy(mutator.arch),
            })

        except Exception:
            pass

    return candidates[:n_candidates]
