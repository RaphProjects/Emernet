import torch
from .base import Module, ModuleType, MappingType
import random

class Activation(Module):
    _mapping_type = MappingType.MAPPER
    def __init__(self, sharpness = 1, symmetry = 1, gate = 0, learnable = False, name = None):
        super().__init__(name, ModuleType.ACTIVATION)
        self.learnable = learnable
        if learnable:
            self.raw_sharpness = torch.nn.Parameter(torch.tensor(sharpness))
            self.raw_symmetry = torch.nn.Parameter(torch.tensor(symmetry))
            self.raw_gate = torch.nn.Parameter(torch.tensor(gate))
        else:
            self.register_buffer("sharpness", torch.sigmoid(torch.tensor(float(sharpness))))
            self.register_buffer("symmetry", torch.sigmoid(torch.tensor(float(symmetry))))
            self.register_buffer("gate", torch.sigmoid(torch.tensor(float(gate))))

    @property
    def mapping_type(self) -> MappingType:
        return MappingType.MAPPER

    @staticmethod
    def random_parameters():
        return [random.random(), random.random(), random.random(), random.random()>=0.5]


    def forward(self, inputs):
        outputs = []
        
        if self.learnable:
            sharpness = torch.sigmoid(self.raw_sharpness)
            symmetry = torch.sigmoid(self.raw_symmetry)
            gate = torch.sigmoid(self.raw_gate)
        else:
            sharpness = self.sharpness
            symmetry = self.symmetry
            gate = self.gate
        for x in inputs:
            base = (1 - sharpness) * torch.tanh(x) + sharpness * torch.relu(x)
            base = (1 - symmetry) * torch.relu(base) + symmetry * base
            base = gate * x * torch.sigmoid(x) + (1 - gate) * base
            outputs.append(base)
        return outputs

    
