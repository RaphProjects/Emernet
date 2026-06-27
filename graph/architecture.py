import networkx
import torch
import pickle

from modules.base import MappingType
from modules.base import ModuleType

class Architecture(networkx.DiGraph):
    def __init__(self):
        super().__init__()
        self.representation_list = []

    def add_node(self, node_id, module):
        super().add_node(node_id,module=module)
    
    def append_node(self, module):
        numeric_ids = [node_id for node_id in self.nodes if isinstance(node_id, int)]
        new_id = (max(numeric_ids) + 1) if numeric_ids else 0
        while new_id in self.nodes:
            new_id += 1
        super().add_node(new_id, module=module)
        return new_id
    

    def add_edge(self, source_id, target_id):
        super().add_edge(source_id, target_id)
    
    def remove_node(self, node_id):
        super().remove_node(node_id)
        for edge in self.edges:
            if edge[0] == node_id or edge[1] == node_id:
                super().remove_edge(edge[0], edge[1])


    def get_Output_id(self):
        for node in self.nodes:
            if len(list(self.successors(node))) == 0:
                return node

    def validation_errors(self):
        errors = []
        if not networkx.is_directed_acyclic_graph(self):
            try:
                cycle = networkx.find_cycle(self)
                cycle_str = " -> ".join(str(edge[0]) for edge in cycle + [cycle[0]])
                errors.append(f"The architecture is not a DAG; cycle: {cycle_str}")
            except Exception:
                errors.append("The architecture is not a DAG")
            return errors

        input_nodes = []
        output_nodes = []
        for node in self.nodes:
            module = self.nodes[node]['module']
            if module.mapping_type == MappingType.SOURCE and len(list(self.successors(node))) == 0:
                errors.append(f"Source node {node} ({module.__class__.__name__}) is not connected to anything")
            if len(list(self.predecessors(node))) == 0:
                if module.mapping_type != MappingType.SOURCE:
                    errors.append(f"Node {node} doesn't have predecessor and is not a source")
                if module.module_type == ModuleType.INPUT:
                    input_nodes.append(node)
            if len(list(self.successors(node))) == 0:
                output_nodes.append(node)

        if len(input_nodes) > 1:
            errors.append(f"There are more than one input nodes: {input_nodes}")
        if len(output_nodes) > 1:
            errors.append(f"There are more than one output nodes: {output_nodes}")
        if len(input_nodes) == 0 or len(output_nodes) == 0:
            errors.append("There are no inputs or outputs")

        if len(output_nodes) == 1:
            output_node = output_nodes[0]
            output_ancestors = networkx.ancestors(self, output_node)
            for node in self.nodes:
                if node not in output_ancestors and node != output_node:
                    errors.append(f"Node {node} is not an ancestor of the output {output_node}")

        return errors

    def isValid(self, verbose=True):
        errors = self.validation_errors()
        if errors:
            if verbose:
                for error in errors:
                    print(error)
            return False
        return True

    def isValid_old(self, verbose=True):
        if not networkx.is_directed_acyclic_graph(self):
            if verbose:
                print("The architecture is not a DAG")
            return False
        
        # Checking if nodes with no predecessors are sources and if there is only one input
        n_inputs = 0
        for node in self.nodes:
            if len(list(self.predecessors(node))) == 0:
                if self.nodes[node]['module'].mapping_type != MappingType.SOURCE: # node doesn't have predecessor and is not a source
                    if verbose:
                        print(f"Node {node} doesn't have predecessor and is not a source")
                    return False
                if self.nodes[node]['module'].module_type == ModuleType.INPUT: # node is an input
                    input_node = node
                    n_inputs += 1
                if n_inputs > 1:
                    if verbose:
                        print(f"There are more than one input nodes")
                    return False


        # Checking if there is only one output
        n_outputs = 0
        for node in self.nodes:
            if len(list(self.successors(node))) == 0:
                n_outputs += 1
                output_node = node
            if n_outputs > 1:
                if verbose:
                    print(f"There are more than one output nodes")
                return False
        
        if n_outputs == 0 or n_inputs == 0:
            if verbose:
                print(f"nodes : {list(self.nodes)}")
                print(f"There are no inputs or outputs")
            return False

        # reachability        
        output_ancestors = networkx.ancestors(self, output_node)
        for node in self.nodes:
            if node not in output_ancestors and node != output_node: # if not every node is an ancestor of the output (the output isn't reachable from every node)
                if verbose:
                    print(f"Node {node} is not an ancestor of the output")
                return False
        
        return True

    def direct_successors(self,target_node):
        direct_successors=[]
        for edge in self.edges:
            if edge[0]==target_node:
                direct_successors.append(edge[1])
        return direct_successors
    
    def direct_ancestors(self,target_node):
        direct_ancestors=[]
        for edge in self.edges:
            if edge[1]==target_node:
                direct_ancestors.append(edge[0])
        return direct_ancestors
    
    def describe(self):
        for node in self.nodes:
            print(f"Node {node} is of type {self.nodes[node]['module'].module_type}")
            print(f"It has {len(list(self.predecessors(node)))} predecessors")
            print("Its direct successors are:")
            for successor in list(self.successors(node)):
                print(f"\t{successor}")


    def parameter_count(self):
        count = 0
        for node in self.nodes:
            if self.nodes[node]['module'].module_type == ModuleType.LEARNABLE:
                count += self.nodes[node]['module'].get_n_parameters()
        return count
    
    def reset_state(self):
        for node in self.nodes:
            self.nodes[node]['module'].reset_state()
    
    def save(self,filepath:str):
        """Saves the architecture to a file using pickle."""
        self.reset_state()

        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"Architecture saved to {filepath}")

    @staticmethod
    def load(filepath: str):
        """Loads an architecture from a file."""
        with open(filepath, 'rb') as f:
            arch = pickle.load(f)
        print(f"Architecture loaded successfully from {filepath}")
        return arch

    def todict(self):
        nodes = [self.nodes[node]['module'].todict() for node in self.nodes]
        edges = self.edges
        return {"nodes": nodes, "edges": edges}
