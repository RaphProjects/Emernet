from .base import Module, ModuleType, MappingType
import torch
import random


class LearnableParameter(Module):
    _mapping_type = MappingType.SOURCE

    def __init__(self, shape, freeze_prob=-1, name=None):
        super().__init__(name, ModuleType.LEARNABLE)
        self.value = torch.nn.Parameter(torch.randn(*shape) * 0.01)
        self.n_parameters = self.value.numel()
        self.freeze_prob = freeze_prob
        self._seed = random.randint(0, 2**31 - 1)

    @property
    def mapping_type(self) -> MappingType:
        return MappingType.SOURCE

    def get_n_parameters(self):
        return self.n_parameters

    def should_update(self, epoch):
        if self.freeze_prob < 0:
            return True
        rng = random.Random(f"{self._seed}_{epoch}")
        return rng.random() < self.freeze_prob

    @staticmethod
    def random_parameters():
        shape = (1, random.choice([2, 4, 8, 16, 32]), random.choice([2, 4, 8, 16, 32]))
        if random.random() < 0.3:
            prob = max(0.0, min(1.0, random.gauss(0.3, 0.1)))
            return [shape, prob]
        return [shape]

    def forward(self, x=None):
        return [self.value]