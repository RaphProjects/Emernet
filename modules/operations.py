from .base import Module, ModuleType, MappingType
import torch



class Add(Module):
    _mapping_type = MappingType.REDUCER
    def __init__(self, name = None):
        super().__init__(name, ModuleType.BASIC)
        self.p_projections = None
        self.f_projections = None
        self.n_parameters = 0

    @property
    def mapping_type(self) -> MappingType:
        return MappingType.REDUCER

    def reset_state(self):
        self.p_projections = None
        self.f_projections = None
        self.n_parameters = 0

    def forward(self, inputTensors):
        
        # Mismatch handling using projections
        if self.p_projections is None:
            '''
            for i, t in enumerate(inputTensors):
                print(f"Add input {i}: shape {t.shape}")
            '''
            self.p_projections = torch.nn.ModuleList()
            self.f_projections = torch.nn.ModuleList()
            # We adapt to the longest vector to avoid loss of information
            tensor_max_f = max(inputTensors, key=lambda t: t.shape[-1])
            for t in inputTensors:
                if t.shape[-1] != tensor_max_f.shape[-1]:
                    linproj = torch.nn.Linear(t.shape[-1], tensor_max_f.shape[-1], bias=True)
                else:
                    linproj = torch.nn.Identity()
                self.f_projections.append(linproj)

            tensor_max_p = max(inputTensors, key=lambda t: t.shape[-2])
            for t in inputTensors:
                if t.shape[-2] != tensor_max_p.shape[-2]:
                    linproj = torch.nn.Linear(t.shape[-2], tensor_max_p.shape[-2], bias=True)
                else:
                    linproj = torch.nn.Identity()
                self.p_projections.append(linproj)

            self.p_projections.to(inputTensors[0].device)
            self.f_projections.to(inputTensors[0].device)
            self.n_parameters += sum(p.numel() for p in self.p_projections.parameters())
            self.n_parameters += sum(p.numel() for p in self.f_projections.parameters())

        # DEBUG - print(f"First input shape = {inputTensors[0].shape}, projection f = {self.f_projections[0]}, projection p = {self.p_projections[0]}")
        summedTensors_f = self.f_projections[0](inputTensors[0])
        summedTensors = (self.p_projections[0](summedTensors_f.transpose(1,2))).transpose(1,2)

        for i in range(1, len(inputTensors)):
            # DEBUG -print(f"{i} of {len(inputTensors)} - sum shape = {summedTensors.shape}, input shape = {inputTensors[i].shape}, projection f = {self.f_projections[i]}, projection p = {self.p_projections[i]}")
            proj_f = self.f_projections[i](inputTensors[i])
            proj_p = (self.p_projections[i](proj_f.transpose(1,2))).transpose(1,2)
            summedTensors = summedTensors + proj_p
        return [summedTensors]
        
class MatMul(Module):
    _mapping_type = MappingType.REDUCER
    def __init__(self, name = None):
        super().__init__(name,  ModuleType.BASIC)
        self.projections = None
        self.n_parameters = 0

    @property
    def mapping_type(self) -> MappingType:
        return MappingType.REDUCER

    def reset_state(self):
        self.projections = None
        self.n_parameters = 0
    
    def forward(self, inputTensors):
        projected_tensors = inputTensors
        mulTensors = inputTensors[0]

        # Mismatch handling using projections
        running_shape_lastdim = mulTensors.shape[-1]
        selfproj_initialised = True
        if self.projections is None:
            self.projections = torch.nn.ModuleList()
            selfproj_initialised = False
            for i in range(1, len(inputTensors)):
                if running_shape_lastdim != inputTensors[i].shape[-2]:
                    self.projections.append(torch.nn.Linear(running_shape_lastdim, inputTensors[i].shape[-2], bias=True))
                else:
                    self.projections.append(torch.nn.Identity())
                running_shape_lastdim = inputTensors[i].shape[-1]
            self.projections.to(inputTensors[0].device)
            self.n_parameters += sum(p.numel() for p in self.projections.parameters())

        # Main loop
        for i in range(1,len(inputTensors)):
            # DEBUG - print(f"Iterating on {i}, running shape: {mulTensors.shape}, input shape: {inputTensors[i].shape}, projection: {self.projections[i-1]}")
            mulTensors = torch.matmul(self.projections[i-1](mulTensors), inputTensors[i])

        return [mulTensors]