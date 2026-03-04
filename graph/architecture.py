import networkx
import torch

from modules.base import MappingType
from modules.base import ModuleType

class Architecture(networkx.DiGraph):
    def __init__(self):
        super().__init__()
        self.representation_list = []

    def add_node(self, node_id,module,):
        super().add_node(node_id,module=module)
    
    def append_node(self,module,):
        super().add_node(len(self.nodes),module=module)
        return len(self.nodes)-1

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

    def isValid(self):
        if not networkx.is_directed_acyclic_graph(self):
            print("The architecture is not a DAG")
            return False
        
        # Checking if nodes with no predecessors are sources and if there is only one input
        n_inputs = 0
        for node in self.nodes:
            if len(list(self.predecessors(node))) == 0:
                if self.nodes[node]['module'].mapping_type != MappingType.SOURCE: # node doesn't have predecessor and is not a source
                    print(f"Node {node} doesn't have predecessor and is not a source")
                    return False
                if self.nodes[node]['module'].module_type == ModuleType.INPUT: # node is an input
                    input_node = node
                    n_inputs += 1
                if n_inputs > 1:
                    print(f"There are more than one input nodes")
                    return False


        # Checking if there is only one output
        n_outputs = 0
        for node in self.nodes:
            if len(list(self.successors(node))) == 0:
                n_outputs += 1
                output_node = node
            if n_outputs > 1:
                print(f"There are more than one output nodes")
                return False
        
        if n_outputs == 0 or n_inputs == 0:
            print(f"nodes : {list(self.nodes)}")
            print(f"There are no inputs or outputs")
            return False

        # reachability        
        output_ancestors = networkx.ancestors(self, output_node)
        for node in self.nodes:
            if node not in output_ancestors and node != output_node: # if not every node is an ancestor of the output (the output isn't reachable from every node)
                print(f"Node {node} is not an ancestor of the output") 
                return False
        
        return True

    def describe(self):
        for node in self.nodes:
            print(f"Node {node} is of type {self.nodes[node]['module'].module_type}")
            print(f"It has {len(list(self.predecessors(node)))} predecessors")
            print("Its direct successors are:")
            for successor in list(self.successors(node)):
                print(f"\t{successor}")
    

    def description(self):
        desc = ""
        for node in self.nodes:
            desc += f"Node {node} is of type {self.nodes[node]['module'].module_type}"
            desc+=f"It has {len(list(self.predecessors(node)))} predecessors"
            desc+="Its direct successors are:"
            for successor in list(self.successors(node)):
                desc+=f"\t{successor}"
        return desc


    def parameter_count(self):
        count = 0
        for node in self.nodes:
            if self.nodes[node]['module'].module_type == ModuleType.LEARNABLE:
                count += self.nodes[node]['module'].get_n_parameters()
        return count
