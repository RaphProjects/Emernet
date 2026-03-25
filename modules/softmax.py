from .base import Module, ModuleType, MappingType
import torch
import random

class SoftMax(Module):
    _mapping_type = MappingType.MAPPER
    def __init__(self, dimension=2, name = None):
        super().__init__(name, ModuleType.SOFTMAX)
        self.dim = dimension
        self.n_parameters = 0

    @property
    def mapping_type(self) -> MappingType:
        return MappingType.MAPPER
    
    @staticmethod
    def random_parameters():
        return None
    
    def reset_state(self):
        self.n_parameters = 0

    def forward(self, inputTensors):
        return [torch.nn.functional.softmax(t, dim=self.dim) for t in inputTensors]