from .base import Module, ModuleType, MappingType
import torch
import random

class Data(Module):
    _mapping_type = MappingType.SOURCE
    def __init__(self,shape, name = None):
        super().__init__(name, ModuleType.DATA)
        self.value = torch.randn(*shape)*0.01
        self.n_parameters = 0

    @property
    def mapping_type(self) -> MappingType:
        return MappingType.SOURCE
    
    def get_n_parameters(self):
        return 0
    
    @staticmethod
    def random_parameters():
        return [(1,random.choice([2, 4, 8, 16, 32]), random.choice([2, 4, 8, 16, 32]))]
    
    def forward(self, x=None):
        return [self.value]