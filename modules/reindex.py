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
        self.index_map = None 
        self.p_out = None
        self.n_parameters = 0

    def initialize_params(self, input_size):
        """Seed the mapping on the first forward pass based on input shape."""
        if self.input_size is None:
            self.input_size = input_size

    def initialize_routing(self, P_out, p_unique=0.8, p_collision=0.1, p_truncate=0.1):
        """
        Hard-codes the routing strategy deterministically.
        This defines how the graph will behave for its entire lifetime.
        """
        P_in = self.input_size[1]
        
        # Ensure probabilities sum to 1
        total = p_unique + p_collision + p_truncate
        p_unique /= total
        p_collision /= total
        p_truncate /= total

        rand_val = random.random()

        if rand_val < p_unique:     # 80% - Unique Mapping
            self._initialize_unique_map(P_in, P_out)
        elif rand_val < p_unique + p_collision:  # 10% - Collision Map
            self._initialize_collision_map(P_in, P_out)
        else:                       # 10% - Truncation Map
            self._initialize_truncate_map(P_in, P_out)

        self.initialized = True

    def _initialize_unique_map(self, P_in, P_out):
        """Assigns each input position to a unique output position."""
        if P_in > P_out:
            # Truncate (take first P_out)
            self.routing_map = np.arange(P_out)
        else:
            # Pad with zeros (duplicates last element)
            self.routing_map = np.zeros(P_out, dtype=int)
            self.routing_map[:P_in] = np.arange(P_in)

    def _initialize_collision_map(self, P_in, P_out):
        """Maps multiple input positions to the same output position."""
        self.routing_map = np.zeros(P_out, dtype=int)
        for i in range(P_out):
            # For each output position, randomly pick an input position to copy from
            self.routing_map[i] = random.randint(0, P_in - 1)

    def _initialize_truncate_map(self, P_in, P_out):
        """Truncates the input sequence (drops the end)."""
        self.routing_map = np.arange(P_out)
        
    def forward(self, inputTensors):
        X = inputTensors[0]
        B, P_in, F = X.shape

        # Deterministic initialization on first run
        if not self.initialized:
            # Decision: 80% Unique, 10% Collision, 10% Truncation
            # Try to make P_out = P_in initially, but allow deviation.
            if random.random() < 0.8:
                P_out = P_in
            elif random.random() < 0.5:
                P_out = P_in + random.randint(1, 3)
            else:
                P_out = max(1, P_in - random.randint(1, 3))
                
            self.initialize_routing(P_out)
            self.initialized = True
            
        P_out = len(self.routing_map)

        # Apply routing
        X_reindexed = X[:, self.routing_map, :]
        
        return [X_reindexed]