from .base import Module, ModuleType, MappingType
import torch
import random

class Concat(Module):
    _mapping_type = MappingType.REDUCER
    def __init__(self, dimension = 2, name = None):
        super().__init__(name, ModuleType.STRUCTURAL)
        self.projections = None
        self.n_parameters = 0
        self.dimension = dimension

    @property
    def mapping_type(self) -> MappingType:
        return MappingType.REDUCER
    
    @staticmethod
    def random_parameters():
        return [random.choice([1, 2])]
    
    def reset_state(self):
        self.projections = None
        self.n_parameters = 0

    def forward(self, inputTensors):
        
        # Mismatch handling using projections
        if self.projections is None:
            '''
            for i, t in enumerate(inputTensors):
                print(f"Add input {i}: shape {t.shape}")
            '''
            dim_to_match = 2 if self.dimension == 1 else 1
            
            self.projections = torch.nn.ModuleList()
            # We adapt to the longest vector to avoid loss of information
            max_size = max(inputTensors, key=lambda t: t.shape[dim_to_match]).shape[dim_to_match]
            for t in inputTensors:
                if t.shape[dim_to_match] != max_size:
                    self.projections.append(torch.nn.Linear(t.shape[dim_to_match], max_size, bias=True))
                else:
                    self.projections.append(torch.nn.Identity())

            self.projections.to(inputTensors[0].device)
            self.n_parameters = sum(p.numel() for p in self.projections.parameters())

        projected_tensors = []
        if self.dimension == 1:
            for i,t in enumerate(inputTensors): # we adapt the dim 2, no need to transpose
                projected_tensors.append(self.projections[i](t))
        else:
            for i,t in enumerate(inputTensors):
                t = t.transpose(1,2)
                adapted_t = self.projections[i](t).transpose(1,2)
                projected_tensors.append(adapted_t)

        return [torch.cat(projected_tensors, dim=self.dimension)]
    
class Split(Module):
    _mapping_type = MappingType.MAPPER
    def __init__(self, dimension = 2, fraction = 0.5, name = None):
        super().__init__(name, ModuleType.STRUCTURAL)
        self.n_parameters = 0
        self.dimension = dimension
        self.fraction = fraction

    @property
    def mapping_type(self) -> MappingType:
        return MappingType.MAPPER
    
    @staticmethod
    def random_parameters():
        return [random.choice([1, 2]), random.uniform(0.0, 1.0)]
    
    def reset_state(self):
        self.n_parameters = 0

    def forward(self, inputTensors):
        output_tensors = []
        

        for i,t in enumerate(inputTensors): 
            dim_size = t.shape[self.dimension]
            if dim_size <= 1:
                # Fallback: if we can't split, we just duplicate it into two branches
                output_tensors.append(t)
                output_tensors.append(t)
            else:
                split_idx = int(dim_size * self.split_ratio)
                split_idx = max(1, min(dim_size - 1, split_idx)) # safety first

                part1, part2 = torch.split(t, [split_idx, dim_size - split_idx], dim=self.dimension)

                output_tensors.append(part1)
                output_tensors.append(part2)



        return output_tensors

class Shift(Module):
    _mapping_type = MappingType.MAPPER
    def __init__(self, amount, dim=1, name = None):
        super().__init__(name, ModuleType.STRUCTURAL)
        self.n_parameters = 0
        self.amount = amount
        self.dim = dim

    @property
    def mapping_type(self) -> MappingType:
        return MappingType.MAPPER
    
    @staticmethod
    def random_parameters():
        return [random.choice([-1,+1]), random.choice([1,1,1,1,1,2])]
    
    def reset_state(self):
        self.n_parameters = 0

    def forward(self, inputTensors):

        return [torch.roll(t, shifts=self.amount, dims=self.dim) for t in inputTensors]
    
class Transpose(Module):
    _mapping_type = MappingType.MAPPER
    def __init__(self, name = None):
        super().__init__(name, ModuleType.STRUCTURAL)
        self.n_parameters = 0

    @property
    def mapping_type(self) -> MappingType:
        return MappingType.MAPPER
    
    @staticmethod
    def random_parameters():
        return None
    
    def reset_state(self):
        self.n_parameters = 0

    def forward(self, inputTensors):
        output_tensors = []       

        for i,t in enumerate(inputTensors): 
            output_tensors.append(t.transpose(1,2))

        return output_tensors