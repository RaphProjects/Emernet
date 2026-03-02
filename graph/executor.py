import networkx
import torch
from torch.utils.data import TensorDataset, DataLoader
from .architecture import Architecture
from modules.activations import *
from modules.aggregation import *
from modules.base import ModuleType
from modules.input import *
from modules.learnable import *
from modules.memory import *
from modules.operations import *

class Executor(torch.nn.Module):
    def __init__(self, architecture : Architecture):
        super().__init__()
        # Check if architecture is valid
        if not architecture.isValid():
            raise Exception("The architecture is not valid")
        
        self.architecture = architecture
        self.output_f_linproj = None
        self.output_p_linproj = None
        for node in architecture.nodes:
            module = architecture.nodes[node]['module']
            self.add_module(str(node), module)

        self.output_node = architecture.get_Output_id()
        self.adapter = False
    
    def set_Output_Adapter(self, input, target_shape, force=False):
        if not self.adapter or force:
            
            target_shape = list(target_shape)
            while len(target_shape)<3:# convert the input shape to a 3D tensor
                target_shape.insert(0,1)
            with torch.no_grad():
                output_shape = self.forward(input, verbose=False)[0].shape
            #print(f"the output shape is {output_shape} (sent by forward), but the target shape is {target_shape}")
            self.output_f_linproj = torch.nn.Linear(output_shape[-1], target_shape[-1])
            self.output_p_linproj = torch.nn.Linear(output_shape[1], target_shape[1])
            #print(f"output_f_linproj shape: {self.output_f_linproj.weight.shape}, output_p_linproj shape: {self.output_p_linproj.weight.shape}")

            self.output_p_linproj.to(input.device)
            self.output_f_linproj.to(input.device)

            self.adapter = True
    
    def output_adapter(self, input):
        input_f_adapted = self.output_f_linproj(input)
        input_f_adapted_t = input_f_adapted.transpose(1,2)
        input_p_adapted = self.output_p_linproj(input_f_adapted_t)
        input_p_adapted_t = input_p_adapted.transpose(1,2)
        return input_p_adapted_t



    def forward(self, input : torch.Tensor, current_node = None, cache_dict = None, verbose=False):
        """
            returns the output of the current node
        """
        if cache_dict == None:
            cache_dict = {}
        
        if current_node == None:
            current_node = self.output_node


        # get the direct predecessors
        direct_predecessors = sorted(list(self.architecture.predecessors(current_node)))

        if len(direct_predecessors) == 0: # source node case
            if self.architecture.nodes[current_node]['module'].mapping_type == MappingType.SOURCE:
                if self.architecture.nodes[current_node]['module'].module_type == ModuleType.INPUT: # input node case
                    self.architecture.nodes[current_node]['module'].set_data(input)
                    out = self.architecture.nodes[current_node]['module'].forward()
                    if verbose:
                        print(f"Node {current_node} is an input node, output shape is {out[0].shape}")
                    return out
                else:
                    out = self.architecture.nodes[current_node]['module'].forward()  
                    if verbose:
                        print(f"Node {current_node} is not source not input, output shape is {out[0].shape}")
                    return self.architecture.nodes[current_node]['module'].forward()  
            
            else: # ERROR non-source node with no direct predecessors
                raise Exception(f"The node {current_node} is not a source, but has no direct predecessors")

        else:
            input_tensors = []
            for predecessor in direct_predecessors:
                if predecessor not in cache_dict:
                    cache_dict[predecessor] = self.forward(input=input, current_node=predecessor, cache_dict=cache_dict, verbose=verbose) # recursive call
                input_tensors.extend(cache_dict[predecessor])
            if current_node == self.output_node:
                raw_outputs =  self.architecture.nodes[current_node]['module'].forward(input_tensors)
                if self.output_f_linproj == None:
                    '''
                    for raw in raw_outputs:
                        print(f"raw output shape, {raw.shape}")
                    print(f"sending the element with the largest batch for our adapter")
                    '''
                    best_output = max(raw_outputs, key=lambda t: t.shape[0])
                    # print(f"best output shape, {best_output.shape}")
                    return [best_output]
                
                 
                best_output = max(raw_outputs, key=lambda t: t.shape[0])
                adapted_output = self.output_adapter(best_output)
                return [adapted_output]
            
            # Not an output node nor a source node
            out = self.architecture.nodes[current_node]['module'].forward(input_tensors)
            if verbose:
                print(f"Node {current_node} is neither an output node nor a source node, output shape is {out[0].shape}")
            return out
    

    def randomize_weights(self):
        for param in self.parameters():
            torch.nn.init.normal_(param, mean=0, std=0.01)


    def fit(self, input, target, verbose=False, lr=0.001, max_iter=600, batch_size=16, patience = 10, min_delta = 1e-7, device = None, cpu = False):
        executor = self
        #executor.set_Output_Adapter(input, target.shape)
        #output = executor.forward(input)
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() and not cpu else 'cpu')
        executor = executor.to(device)
        
        executor.set_Output_Adapter(input[:batch_size].to(device), target.shape)
        optimizer = torch.optim.Adam(executor.parameters(), lr=lr) 
        best_loss = float('inf')

        wait = 0
        stop_training = False

        dataset = TensorDataset(input, target)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        for epoch in range(max_iter):
            epoch_loss = 0
            for batch_input, batch_target in dataloader:
                batch_input = batch_input.to(device)
                batch_target = batch_target.to(device)
                output = executor.forward(batch_input)
                loss = torch.nn.functional.mse_loss(output[0], batch_target)

                
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                epoch_loss += loss.item()
    
            avg_loss = epoch_loss / len(dataloader)
            if verbose and epoch % 5 == 0:
                    print(f"Loss: {loss.item()}, epoch {epoch}/{max_iter}")

            # Early stopping
            if avg_loss < best_loss-min_delta:
                best_loss = avg_loss
                wait = 0
            else:
                wait += 1
                if wait == patience:
                    break

            


    