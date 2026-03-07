import torch
from .base import Module, ModuleType, MappingType

class Input(Module):
    _mapping_type = MappingType.SOURCE
    def __init__(self, name=None):
        super().__init__(name, ModuleType.INPUT)
        
    def get_n_parameters(self):
        return 0

    @property
    def mapping_type(self) -> MappingType:
        return MappingType.SOURCE
    

    def forward(self, x=None):
        return [self.StoredData]
    
    def set_data(self, data):
        self.StoredData = data
    
