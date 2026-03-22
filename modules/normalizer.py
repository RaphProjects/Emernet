from .base import Module, ModuleType, MappingType
import torch
import random

class Normalizer(Module):
    _mapping_type = MappingType.MAPPER
    def __init__(self, dimension = 0, affine = True, name = None):
        super().__init__(name, ModuleType.NORM)
        self.n_parameters = 0
        self.affine = affine
        self.dimension = dimension
        self.norm_layers = None

    @property
    def mapping_type(self) -> MappingType:
        return MappingType.MAPPER
    
    def get_n_parameters(self):
        return self.n_parameters
    
    @staticmethod
    def random_parameters():
        return [random.choice([0,1,2]), random.random()>=0.5]

    def reset_state(self):
        self.norm_layers = None
        self.n_parameters = 0

    def forward(self, inputTensors):
        outputs = []
        if self.norm_layers is None:
            self.norm_layers = torch.nn.ModuleList()
            for tensor in inputTensors:
                if self.dimension == 0:
                    self.norm_layers.append(torch.nn.LayerNorm(tensor.shape[-1], elementwise_affine=self.affine))
                if self.dimension == 1:
                    self.norm_layers.append(torch.nn.BatchNorm1d(tensor.shape[-1], affine=self.affine))
                if self.dimension == 2:
                    self.norm_layers.append(torch.nn.InstanceNorm1d(tensor.shape[-1], affine=self.affine))

            self.norm_layers.to(inputTensors[0].device)
            self.n_parameters += sum(p.numel() for p in self.norm_layers.parameters())
        
        for tensor, norm_layer in zip(inputTensors, self.norm_layers):
            if self.dimension == 0:
                tensor = norm_layer(tensor)
            elif self.dimension == 1:
                tensor = tensor.transpose(1,2)
                tensor = norm_layer(tensor)
                tensor = tensor.transpose(1,2)
            elif self.dimension == 2:
                tensor = tensor.transpose(1,2)
                tensor = norm_layer(tensor)
                tensor = tensor.transpose(1,2)
            outputs.append(tensor)
        return outputs