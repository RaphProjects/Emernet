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
        self.output_f_linproj = torch.nn.Identity() # Dummies
        self.output_p_linproj = torch.nn.Identity()
        '''
        self.output_f_linproj = None
        self.output_p_linproj = None
        '''
        self.output_index = 0
        for node in architecture.nodes:
            module = architecture.nodes[node]['module']
            self.add_module(str(node), module)

        self.output_node = architecture.get_Output_id()
        self.adapter = False

    def pick_output(self, raw_outputs, target_shape=None):
        indexed = list(enumerate(raw_outputs))
        max_batch = max(t.shape[0] for _, t in indexed)
        candidates = [(i, t) for i, t in indexed if t.shape[0] == max_batch]

        if target_shape is None:
            return candidates[0][1], candidates[0][0]

        target_p = target_shape[1]
        target_f = target_shape[2]

        best_index, best_tensor = min(
            candidates,
            key=lambda it: abs(it[1].shape[1] - target_p) + abs(it[1].shape[2] - target_f)
        )
        return best_tensor, best_index

    
    def set_Output_Adapter(self, input, target_shape, force=False):
        if not self.adapter or force:
            
            target_shape = list(target_shape)
            while len(target_shape)<3:# convert the input shape to a 3D tensor
                target_shape.insert(0,1)
            with torch.no_grad():
                raw_outputs = self.forward(input, verbose=False, adapting=True)
                chosen, chosen_index = self.pick_output(raw_outputs, target_shape)
                chosen_shape = chosen.shape
            #print(f"the output shape is {output_shape} (sent by forward), but the target shape is {target_shape}")
            self.output_f_linproj = torch.nn.Linear(chosen_shape[-1], target_shape[-1],device=input.device)
            self.output_p_linproj = torch.nn.Linear(chosen_shape[1], target_shape[1],device=input.device)
            #print(f"output_f_linproj shape: {self.output_f_linproj.weight.shape}, output_p_linproj shape: {self.output_p_linproj.weight.shape}")

            self.output_p_linproj.to(input.device)
            self.output_f_linproj.to(input.device)

            self.output_index = chosen_index
            self.adapter = True
    
    def output_adapter(self, input):
        input_f_adapted = self.output_f_linproj(input)
        input_f_adapted_t = input_f_adapted.transpose(1,2)
        input_p_adapted = self.output_p_linproj(input_f_adapted_t)
        input_p_adapted_t = input_p_adapted.transpose(1,2)
        return input_p_adapted_t



    def forward(self, input : torch.Tensor, current_node = None, cache_dict = None, verbose=False, adapting = False):
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
                
                raw_outputs = self.architecture.nodes[current_node]['module'].forward()
                if verbose:
                    print(f"Node {current_node} is a source node, output shape is {raw_outputs[0].shape}")
            
            else: # ERROR non-source node with no direct predecessors
                raise Exception(f"The node {current_node} is not a source, but has no direct predecessors")

        else: # not a source node
            input_tensors = []
            for predecessor in direct_predecessors:
                if predecessor not in cache_dict:
                    cache_dict[predecessor] = self.forward(input=input, current_node=predecessor, cache_dict=cache_dict, verbose=verbose) # recursive call
                input_tensors.extend(cache_dict[predecessor])

            raw_outputs =  self.architecture.nodes[current_node]['module'].forward(input_tensors)
            if verbose:
                print(f"Node {current_node} is not a source node, output shape is {raw_outputs[0].shape}")

        # if this is the output node, we need to adapt the output to match the target shape
        # this applies regardless of whether the output node is a source node or not
        if current_node == self.output_node:

            if adapting:# This means we are being called from the set_Output_Adapter method, we need to return the raw outputs
                return raw_outputs
        
            if not self.adapter:                          
                # Forward is called without having set the output adapter (forwarding without fitting)
                best_tensor, best_index = self.pick_output(raw_outputs) # pick_output will handle selecting the biggest batch size
                self.output_index = best_index
                return [best_tensor]
            
            # here we are not adapting nor forwarding without adapting : we are fitting
            best_output = raw_outputs[self.output_index]
            adapted_output = self.output_adapter(best_output)
            return [adapted_output]
            
        
        return raw_outputs # not the output node, just return the raw outputs of this node's module

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
        
        executor.set_Output_Adapter(input[:batch_size].to(device), target.shape, force=True)
        '''
        print(f"adapter flag: {executor.adapter}")
        print(f"f_proj type: {type(executor.output_f_linproj)}")
        print(f"f_proj is None: {executor.output_f_linproj is None}")
        '''

        optimizer = torch.optim.Adam(executor.parameters(), lr=lr) 
        best_loss = float('inf')

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min',       # we want loss to go DOWN
            patience=6,       # wait 6 epochs before reducing
            factor=0.6        # new_lr = old_lr * 0.6
        )

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
                if output[0].shape != batch_target.shape:
                    print("\n=== SHAPE MISMATCH DETECTED ===")
                    print(f"  output shape:  {output[0].shape}")
                    print(f"  target shape:  {batch_target.shape}")
                    print(f"  adapter flag:  {executor.adapter}")
                    print(f"  output_index:  {executor.output_index}")
                    print(f"  f_proj:        {executor.output_f_linproj}")
                    print(f"  p_proj:        {executor.output_p_linproj}")
                    print(f"  input shape:   {batch_input.shape}")
                    
                    # Check what raw outputs look like
                    with torch.no_grad():
                        raw = executor.forward(batch_input, adapting=True)
                        print(f"  raw output count: {len(raw)}")
                        for idx, r in enumerate(raw):
                            marker = " ← selected" if idx == executor.output_index else ""
                            print(f"    raw[{idx}] shape: {r.shape}{marker}")
                    
                    print(f"  architecture:")
                    executor.architecture.describe()
                    print("=== END DIAGNOSTIC ===\n")
                    
                    # Skip this batch to avoid crash
                    continue
                loss = torch.nn.functional.mse_loss(output[0], batch_target)

                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(executor.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                epoch_loss += loss.item()
    
            avg_loss = epoch_loss / len(dataloader)
            scheduler.step(avg_loss)
            if verbose and epoch % 5 == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    print(f"Epoch {epoch}/{max_iter} | Loss: {avg_loss:.6f} | LR: {current_lr:.6f}")

            # Early stopping
            if avg_loss < best_loss-min_delta:
                best_loss = avg_loss
                wait = 0
            else:
                wait += 1
                if wait == patience:
                    break

            


    