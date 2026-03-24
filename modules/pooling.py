from .base import Module, ModuleType, MappingType
import torch
import random

class Pooling(Module):
    _mapping_type = MappingType.MAPPER
    def __init__(self, dimension = 2, strategy="mean", name = None):
        super().__init__(name, ModuleType.STRUCTURAL)
        self.n_parameters = 0
        self.dimension = dimension
        self.strategy = strategy

    @property
    def mapping_type(self) -> MappingType:
        return MappingType.MAPPER
    
    @staticmethod
    def random_parameters():
        return [random.choice([1, 2]), random.choice(["mean","median","min","max"])]
    
    def reset_state(self):
        self.n_parameters = 0

    def forward(self, inputTensors):
        output_tensors = []
        
        if self.strategy=="mean":
            for i,t in enumerate(inputTensors): 
                output_tensors.append(torch.mean(t,dim=self.dimension, keepdim=True).values)
        if self.strategy=="min":
            for i,t in enumerate(inputTensors): 
                output_tensors.append(torch.min(t,dim=self.dimension, keepdim=True).values)
        if self.strategy=="max":
            for i,t in enumerate(inputTensors): 
                output_tensors.append(torch.max(t,dim=self.dimension, keepdim=True).values)
        if self.strategy=="median":
            for i,t in enumerate(inputTensors): 
                output_tensors.append(torch.median(t,dim=self.dimension, keepdim=True).values)
        return output_tensors