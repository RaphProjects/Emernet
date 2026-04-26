import torch
import torch.nn.functional as F
from .base import Module, ModuleType, MappingType

class Einsum(Module):
    """
    Pure index contraction for 3D tensors (Batch, Positions, Features).
    This module handles dimension mismatches by automatically padding 
    smaller dimensions to match the largest ones in the group.
    """
    _mapping_type = MappingType.REDUCER

    def __init__(self, input_dims: list, output_dims: list, name=None):
        """
        Args:
            input_dims: A list of lists of integers (e.g., [[3, 4]]).
                        Standard mapping: 0=Batch, 1=Positions, 2=Features.
            output_dims: A list of integers defining the output tensor shape.
        """
        super().__init__(name, ModuleType.BASIC)
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.n_parameters = 0  # Operation nodes must remain pure [5]

    def forward(self, inputTensors):
        # 1. Track the maximum size for each unique dimension label
        # This ensures we always expand to the largest available size [1]
        max_sizes = {}
        for tensor, dims in zip(inputTensors, self.input_dims):
            for i, dim_label in enumerate(dims):
                current_size = tensor.shape[i]
                max_sizes[dim_label] = max(max_sizes.get(dim_label, 0), current_size)

        # 2. Apply zero-padding to align all tensors
        # Padding with zeros is mathematically neutral for contractions (0 * x = 0) [2]
        padded_tensors = []
        for tensor, dims in zip(inputTensors, self.input_dims):
            padding_config = []
            # F.pad expects pairs (left, right) starting from the last dimension
            for dim_label in reversed(dims):
                target_size = max_sizes[dim_label]
                current_size = tensor.shape[dims.index(dim_label)]
                # Add zeros only to the end of the dimension (right-padding)
                padding_config.extend([0, target_size - current_size])
            
            if sum(padding_config) > 0:
                padded_tensors.append(F.pad(tensor, tuple(padding_config)))
            else:
                padded_tensors.append(tensor)

        # 3. Prepare arguments for torch.einsum in numerical format
        # Format: einsum(tensor1, dims1, tensor2, dims2, ..., output_dims)
        einsum_args = []
        for tensor, dims in zip(padded_tensors, self.input_dims):
            einsum_args.extend([tensor, dims])
        einsum_args.append(self.output_dims)

        # Returns the result as a list to remain compatible with Architecture executor
        return [torch.einsum(*einsum_args)]

    def reset_state(self):
        """No internal state to reset for pure operation nodes."""
        pass