import torch
import torch.nn.functional as F
from .base import Module, ModuleType, MappingType
import random
from itertools import permutations

class EinsteinAggregator(Module):
    """
    Pure index contraction for 3D tensors (Batch, Positions, Features).
    Handles dimension mismatches by zero-padding smaller dimensions to match
    the largest ones in the group.
    """
    _mapping_type = MappingType.REDUCER
    _cache = {}   # (seed, n_inputs) -> (notation, T)
    _order_cache = {}  # notation/dim layout -> elimination order

    def __init__(self, name=None):
        super().__init__(name, ModuleType.BASIC)
        self.notation = None
        self.T = None
        self.n_parameters = 0
        # Unique seed that survives deep copies, ensuring identical re‑creation.
        self._seed = random.randint(0, 2**31 - 1)

    @property
    def mapping_type(self) -> MappingType:
        return MappingType.REDUCER

    def init_notation(self, n_inputs):
        """Build the einsum-like notation string. Uses a class cache and a
        per‑instance seed to guarantee that all copies of the same module
        produce the same contraction pattern.
        Capped at 6 unique symbols (a–f) to prevent OOM from the
        high‑dimensional joint tensor."""
        cache_key = (self._seed, n_inputs)
        if cache_key in EinsteinAggregator._cache:
            self.notation, self.T = EinsteinAggregator._cache[cache_key]
            return

        MAX_UNIQUE = 6  # a–f → joint tensor at most (max_dim)^6
        rng = random.Random(self._seed)

        input_notation_list = ["abc"]
        used_symbols = {"a", "b", "c"}

        for i in range(1, n_inputs):
            tensor_symbols = ["a"]
            for dim in range(2):
                if len(used_symbols) < MAX_UNIQUE and rng.random() < 0.3:
                    new_symbol = chr(ord(max(used_symbols)) + 1)
                    tensor_symbols.append(new_symbol)
                    used_symbols.add(new_symbol)
                else:
                    available = [s for s in used_symbols if s not in tensor_symbols and s != "a"]
                    if available:
                        tensor_symbols.append(rng.choice(available))
                    else:
                        tensor_symbols.append(rng.choice(list(used_symbols - {"a"})))
            input_notation_list.append("".join(tensor_symbols))

        output_notation = "a"
        available_for_output = list(used_symbols - {"a"})
        rng.shuffle(available_for_output)

        for _ in range(2):
            if rng.random() > 0.1 and available_for_output:
                output_notation += available_for_output.pop()

        self.notation = ",".join(input_notation_list) + "->" + output_notation
        self.T = max(1e-6, 1.0 + rng.gauss(0, 1.33))

        EinsteinAggregator._cache[cache_key] = (self.notation, self.T)

    @staticmethod
    def _numel_for_symbols(symbols, symbol_to_maxdim):
        n = 1
        for s in symbols:
            n *= symbol_to_maxdim[s]
        return n

    @staticmethod
    def _ordered_union(symbol_groups, canonical_symbols):
        return [
            s for s in canonical_symbols
            if any(s in group for group in symbol_groups)
        ]

    @staticmethod
    def _align_factor(tensor, symbols, target_symbols):
        """Return tensor with axes ordered/broadcastable to target_symbols."""
        symbols = tuple(symbols)
        target_symbols = tuple(target_symbols)

        if symbols:
            perm = [symbols.index(s) for s in target_symbols if s in symbols]
            if perm != list(range(len(perm))):
                tensor = tensor.permute(*perm)

        shape_idx = 0
        reshape_target = []
        for s in target_symbols:
            if s in symbols:
                reshape_target.append(tensor.shape[shape_idx])
                shape_idx += 1
            else:
                reshape_target.append(1)

        if not reshape_target:
            return tensor
        return tensor.reshape(reshape_target)

    def _choose_elimination_order(self, factor_symbols, reduce_symbols, canonical_symbols, symbol_to_maxdim):
        if not reduce_symbols:
            return []

        cache_key = (
            tuple(tuple(symbols) for symbols in factor_symbols),
            tuple(reduce_symbols),
            tuple((s, symbol_to_maxdim[s]) for s in canonical_symbols),
        )
        if cache_key in EinsteinAggregator._order_cache:
            return list(EinsteinAggregator._order_cache[cache_key])

        def estimate_peak(order):
            current = [tuple(symbols) for symbols in factor_symbols]
            peak = 0

            for symbol in order:
                involved = [symbols for symbols in current if symbol in symbols]
                if not involved:
                    continue

                combined = self._ordered_union(involved, canonical_symbols)
                peak = max(peak, self._numel_for_symbols(combined, symbol_to_maxdim))
                reduced = tuple(s for s in combined if s != symbol)
                current = [symbols for symbols in current if symbol not in symbols]
                current.append(reduced)

            final_symbols = self._ordered_union(current, canonical_symbols)
            peak = max(peak, self._numel_for_symbols(final_symbols, symbol_to_maxdim))
            return peak

        best_order = None
        best_peak = None
        for order in permutations(reduce_symbols):
            peak = estimate_peak(order)
            if best_peak is None or peak < best_peak:
                best_order = order
                best_peak = peak

        EinsteinAggregator._order_cache[cache_key] = tuple(best_order)
        return list(best_order)

    def _mean_fallback(self, inputTensors):
        acc = None
        for t in inputTensors:
            reduced = t.mean(dim=tuple(range(1, t.dim())), keepdim=True)
            if acc is None:
                acc = reduced.clone()
            else:
                n = min(acc.shape[0], reduced.shape[0])
                acc[:n] += reduced[:n]
        acc /= len(inputTensors)
        return acc

    def _contract_log_factors(self, factors, output_side, unique_symbols, symbol_to_maxdim, max_intermediate_elems):
        """Exact variable-elimination contraction in log-space.

        This computes the same result as building the full joint tensor and
        applying T * logsumexp(joint / T), but it avoids materialising symbols
        that can be eliminated locally first.
        """
        T = self.T
        reduce_symbols = [s for s in unique_symbols if s not in output_side]
        order = self._choose_elimination_order(
            [symbols for _, symbols in factors],
            reduce_symbols,
            unique_symbols,
            symbol_to_maxdim,
        )

        current_factors = [(tensor, tuple(symbols)) for tensor, symbols in factors]

        for symbol in order:
            involved = [(tensor, symbols) for tensor, symbols in current_factors if symbol in symbols]
            if not involved:
                continue

            combined_symbols = self._ordered_union(
                [symbols for _, symbols in involved],
                unique_symbols,
            )

            if self._numel_for_symbols(combined_symbols, symbol_to_maxdim) > max_intermediate_elems:
                return None

            combined = None
            for tensor, symbols in involved:
                aligned = self._align_factor(tensor, symbols, combined_symbols)
                combined = aligned if combined is None else combined + aligned

            reduce_dim = combined_symbols.index(symbol)
            reduced = T * torch.logsumexp(combined / T, dim=reduce_dim)
            reduced_symbols = tuple(s for s in combined_symbols if s != symbol)

            current_factors = [
                (tensor, symbols)
                for tensor, symbols in current_factors
                if symbol not in symbols
            ]
            current_factors.append((reduced, reduced_symbols))

        result_symbols = [s for s in unique_symbols if s in output_side]
        if self._numel_for_symbols(result_symbols, symbol_to_maxdim) > max_intermediate_elems:
            return None

        result = None
        for tensor, symbols in current_factors:
            aligned = self._align_factor(tensor, symbols, result_symbols)
            result = aligned if result is None else result + aligned

        return result

    def forward(self, inputTensors):
        # Delayed initialisation: notation is built on first use.
        if self.notation is None:
            self.init_notation(len(inputTensors))

        input_side, output_side = self.notation.split("->")
        tensor_notations = input_side.split(",")

        # Determine the maximum size each dimension label reaches across all inputs.
        symbol_to_maxdim = {}
        for tensor, notation in zip(inputTensors, tensor_notations):
            for j, symbol in enumerate(notation):
                symbol_to_maxdim[symbol] = max(
                    symbol_to_maxdim.get(symbol, 0), tensor.shape[j]
                )

        # Zero-pad on the right so every tensor has the same length for shared symbols.
        padded_tensors = []
        for tensor, notation in zip(inputTensors, tensor_notations):
            padding_config = []
            for j in range(len(notation) - 1, -1, -1):
                symbol = notation[j]
                target_size = symbol_to_maxdim[symbol]
                current_size = tensor.shape[j]
                padding_config.extend([0, target_size - current_size])

            if any(p != 0 for p in padding_config):
                padded_tensors.append(F.pad(tensor, tuple(padding_config)))
            else:
                padded_tensors.append(tensor)

        # Build a canonical ordering of all symbols appearing in the inputs.
        unique_symbols = []
        for notation in tensor_notations:
            for char in notation:
                if char not in unique_symbols:
                    unique_symbols.append(char)

        MAX_INTERMEDIATE_ELEMS = 50_000_000

        # Keep tensors as local log-space factors and eliminate reduced
        # symbols one by one instead of materialising the full joint tensor.
        factors = []
        for tensor, notation in zip(padded_tensors, tensor_notations):
            assert len(notation) == tensor.dim(), \
                f"Notation '{notation}' ({len(notation)} chars) vs tensor dim {tensor.dim()}"
            factors.append((tensor, tuple(notation)))

        result = self._contract_log_factors(
            factors,
            output_side,
            unique_symbols,
            symbol_to_maxdim,
            MAX_INTERMEDIATE_ELEMS,
        )
        if result is None:
            return [self._mean_fallback(inputTensors)]

        # Reorder the remaining dimensions to match the exact output notation.
        remaining_symbols = [s for s in unique_symbols if s in output_side]
        perm_out = [remaining_symbols.index(char) for char in output_side]

        if perm_out != list(range(len(perm_out))):
            result = result.permute(*perm_out)

        # Force the output to the canonical 3‑D shape (B, P, F).
        if result.dim() == 0:
            return [result.view(1, 1, 1).expand(inputTensors[0].shape[0], 1, 1)]
        
        batch_size = result.shape[0]

        if result.dim() == 1:
            return [result.view(batch_size, 1, 1)]
        elif result.dim() == 2:
            return [result.view(batch_size, result.shape[1], 1)]
        else:
            return [result.view(batch_size, result.shape[1], -1)]
