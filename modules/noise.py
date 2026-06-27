import torch
from .base import Module, ModuleType, MappingType
import random


class Noise(Module):
    selection_weight = 0.05
    _mapping_type = MappingType.MAPPER
    def __init__(self, noise_type="gaussian", scale=0.1, per_tensor=True, name=None):
        super().__init__(name, ModuleType.BASIC)
        self.noise_type = noise_type
        self.scale = scale
        self.per_tensor = per_tensor
        self.n_parameters = 0

    @property
    def mapping_type(self) -> MappingType:
        return MappingType.MAPPER

    @staticmethod
    def random_parameters():
        return [
            random.choice(["gaussian", "uniform", "bernoulli"]),
            random.uniform(0.01, 0.5),
            random.random() >= 0.3,
        ]

    def reset_state(self):
        self.n_parameters = 0

    def forward(self, inputTensors):
        if not self.training:
            return inputTensors

        outputs = []
        for t in inputTensors:
            if self.noise_type == "gaussian":
                noise = torch.randn_like(t) * self.scale
            elif self.noise_type == "uniform":
                noise = (torch.rand_like(t) * 2 - 1) * self.scale
            elif self.noise_type == "bernoulli":
                noise = (torch.rand_like(t) < self.scale).float()
            else:
                noise = 0
            outputs.append(t + noise)
        return outputs
