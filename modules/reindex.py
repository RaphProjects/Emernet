import torch
import torch.nn.functional as F
from .base import Module, ModuleType, MappingType
import random
import numpy as np

class Reindex(Module):
    """
    Universal positional routing module with dynamic scaling.
    Determines output sequence length (P_out) and routing map at initialization.
    Hard-coded probabilities: 80% Unique Map, 10% Collision, 10% Truncation.
    No learned parameters; routing is fixed after the first pass to ensure determinism.
    """
    _mapping_type = MappingType.MAPPER

    def __init__(self, name=None):
        # By default, Reindex is a Mapper.
        super().__init__(name, ModuleType.BASIC)
        self.ps_out = None
        self.n_parameters = 0
        self.routing_maps = []

    def initialize_params(self, input_size):
        """Seed the mapping on the first forward pass based on input shape."""
        if self.input_size is None:
            self.input_size = input_size

    def initialize_routing(self, inputTensors, p_correct=0.5):
        """
        Hard-codes the routing maps deterministically.
        This defines how the graph will behave for its entire lifetime.
        """
        Ps_in = [t.shape[1] for t in inputTensors]
        Ps_out = self.ps_out
        

        for i in range(len(inputTensors)):
            P_in = Ps_in[i]
            P_out = Ps_out[i]
            routing_map = np.zeros(P_in, dtype=int)
            for p in range(P_in):
                p_destination = random.choice(range(P_out))
                if p_destination in routing_map: # Duplicate
                    if random.random() < p_correct: # 50% chance to try to correct duplicate
                        p_destination = random.choice(range(P_out))
                routing_map[p] = p_destination
            self.routing_maps.append(routing_map)

        self.initialized = True

        
    def forward(self, inputTensors):
        X = inputTensors[0]
        B, P_in, F = X.shape

        # Deterministic initialization on first run
        if not self.initialized:
            Ps_out_size = max(1, len(inputTensors) + int(random.gauss(0, 1.33)))
            Ps_out = []
            for _ in range(Ps_out_size):
                Ps_out.append(max(1, P_in + int(random.gauss(0, 1.33))))
            self.ps_out = Ps_out

            self.initialize_routing(inputTensors, Ps_out)
            self.initialized = True
            
        P_out = len(self.routing_map)

        # Apply routing
        X_reindexed = X[:, self.routing_map, :]
        
        return [X_reindexed]