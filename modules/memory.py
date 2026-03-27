from .base import Module, ModuleType, MappingType
import torch
import random

class EMA(Module):
    _mapping_type = MappingType.MAPPER
    def __init__(self, alpha_init=0.0, name=None):
        super().__init__(name, ModuleType.MEMORY)
        # Learnable decay rate (will be mapped to 0-1 via sigmoid)
        self.raw_alpha = torch.nn.Parameter(torch.tensor(alpha_init))
        self.n_parameters = 1

    @property
    def mapping_type(self) -> MappingType:
        return MappingType.MAPPER

    @staticmethod
    def random_parameters():
        return [random.uniform(-2.0, 2.0)] # Init raw_alpha before sigmoid

    def forward(self, inputTensors):
        outputs = []
        alpha = torch.sigmoid(self.raw_alpha) # Bound between 0 and 1
        
        for x in inputTensors:
            B, P, F = x.shape
            h = torch.zeros(B, F, device=x.device)
            y_steps = []
            
            # Sweep across the sequence (Time)
            for t in range(P):
                h = alpha * h + (1 - alpha) * x[:, t, :]
                y_steps.append(h)
                
            # Stack the list of tensors back into (B, P, F)
            y = torch.stack(y_steps, dim=1)
            outputs.append(y)
            
        return outputs