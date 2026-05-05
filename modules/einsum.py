import torch
import torch.nn.functional as F
from .base import Module, ModuleType, MappingType
import random

class Einsum(Module):
    """
    Pure index contraction for 3D tensors (Batch, Positions, Features).
    This module handles dimension mismatches by automatically padding 
    smaller dimensions to match the largest ones in the group [2].
    """
    _mapping_type = MappingType.REDUCER

    def __init__(self, name=None):
        super().__init__(name, ModuleType.BASIC)
        self.notation = None
        self.n_parameters = 0  # Operation nodes must remain pure [1]

    def init_notation(self, n_inputs):
        """
        Initializes the einsum notation string.
        Ensures the batch dimension ('a') remains at index 0 for all tensors.
        """
        # First tensor is always standard (Batch, Positions, Features)
        input_notation_list = ["abc"]
        used_symbols = {"a", "b", "c"}
        
        # Build notation for subsequent input tensors
        for i in range(1, n_inputs):
            tensor_symbols = ["a"] # Protect batch dimension at index 0
            for dim in range(2):
                if random.random() < 0.3:
                    # Reuse an existing dimension label
                    tensor_symbols.append(random.choice(list(used_symbols)))
                else:
                    # Create a new dimension label
                    new_symbol = chr(ord(max(used_symbols)) + 1)
                    tensor_symbols.append(new_symbol)
                    used_symbols.add(new_symbol)
            input_notation_list.append("".join(tensor_symbols))
        
        # Build output notation (always starts with batch 'a')
        output_notation = "a"
        # Potential candidates for P and F (excluding batch)
        available_for_output = list(used_symbols - {"a"})
        random.shuffle(available_for_output)
        
        # 10% chance to skip each remaining slot to allow rank reduction (3D -> 2D/1D)
        for _ in range(2):
            if random.random() > 0.1 and available_for_output:
                output_notation += available_for_output.pop()
            
        self.notation = ",".join(input_notation_list) + "->" + output_notation

    def forward(self, inputTensors):
        if self.notation is None:
            self.init_notation(len(inputTensors))

        input_side, output_side = self.notation.split("->")
        tensor_notations = input_side.split(",")

        # Determine maximum size for each unique dimension label
        symbol_to_maxdim = {}
        for tensor, notation in zip(inputTensors, tensor_notations):
            for j, symbol in enumerate(notation):
                symbol_to_maxdim[symbol] = max(symbol_to_maxdim.get(symbol, 0), tensor.shape[j])
        
        # Apply zero-padding to align all dimensions (Always Expand)
        padded_tensors = []
        for tensor, notation in zip(inputTensors, tensor_notations):
            padding_config = []
            # Reversed loop because F.pad starts from the last dimension
            for j in range(len(notation) - 1, -1, -1):
                symbol = notation[j]
                target_size = symbol_to_maxdim[symbol]
                current_size = tensor.shape[j]
                # Right-padding only
                padding_config.extend([0, target_size - current_size])
            
            if sum(padding_config) > 0:
                padded_tensors.append(F.pad(tensor, tuple(padding_config)))
            else:
                padded_tensors.append(tensor)

        # Execute Einstein contraction
        result = torch.einsum(self.notation, *padded_tensors)

        # Systematic reshape to 3D (B, P, F) for Emernet compatibility [4]
        batch_size = result.shape
        if result.dim() == 1:
            # Only batch remains: (B,) -> (B, 1, 1)
            return [result.view(batch_size, 1, 1)]
        elif result.dim() == 2:
            # Rank reduced: (B, X) -> (B, X, 1)
            return [result.view(batch_size, -1, 1)]
        else:
            # Standard 3D or high-rank (B, P, H, F...): 
            # Keep Batch and second dim, flatten the rest into Features
            return [result.view(batch_size, result.shape[5], -1)]

    def reset_state(self):
        """Reset notation to allow dynamic re-initialization."""
        self.notation = None