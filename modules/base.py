from enum import Enum,auto
from abc import ABC, abstractmethod

import torch

class ModuleType(Enum):
    BASIC = auto()
    ACTIVATION = auto()
    AGGREGATION = auto()
    MEMORY = auto()
    CONCATENATION = auto()
    STRUCTURAL = auto()
    SOFTMAX = auto()
    LEARNABLE = auto()
    NORM = auto()
    INPUT = auto()
    DATA = auto()

class MappingType(Enum):
    REDUCER = auto()
    MAPPER = auto()
    SOURCE = auto()

class AxisType(Enum):
    POSITIONS = auto()
    FEATURES = auto()

class Module(ABC,torch.nn.Module):
    def __init__(self, name, module_type):
        super().__init__()
        self.name = name
        self.module_type = module_type
        self.n_parameters = 0

    @abstractmethod
    def forward(self, inputs : list[torch.Tensor])->list[torch.Tensor]:
        pass

    @property
    @abstractmethod
    def mapping_type(self) -> MappingType:
        pass
    
    @staticmethod
    def random_parameters():
        return []
        
    def todict(self):
        return {"module_type":self.module_type, "Hparameters":None}

    def reset_state(self):
        pass

    def get_n_parameters(self):
        return 0