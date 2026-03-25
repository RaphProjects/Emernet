import random
import networkx
import torch
from graph.architecture import Architecture
from graph.executor import Executor
from modules.input import Input
from modules.operations import *
from modules.base import *
from modules.learnable import *
from modules.activations import *
from modules.normalizer import *
from modules.structural import *
from modules.pooling import *
from modules.softmax import *
class Generator:
    def __init__(self, generation_type = "agnostic"):
        # self.architecture = Architecture()
        self.generation_type = generation_type
        self.available_modules = [MatMul, Add, Activation, LearnableParameter, Normalizer, Mult, Concat,
                                   Split, Pooling, Transpose,SoftMax]

    def generate(self, n_nodes=12, randomize_n_nodes=True)->Architecture:
        generated = False
        iters = 0
        if randomize_n_nodes:
            perturbations = [-3, -2, -1, 0, 1, 2, 3, 4]
            n_nodes+=random.choice(perturbations)
        if self.generation_type == "dense":
            while not generated and iters < 20:
                try:
                    arch = self.generate_dense(n_nodes)
                    test_input = torch.randn(2, 15, 18) # Sanity check
                    ex = Executor(arch)
                    out = ex.forward(test_input)
                    if torch.isfinite(out[0]).all():
                        return arch
                except Exception as e:
                    pass
                iters += 1
        elif self.generation_type == "agnostic":
            while not generated:
                try:
                    arch = self.generate_order_agnostic(n_nodes)
                    test_input = torch.randn(2, 15, 18)
                    ex = Executor(arch)
                    out = ex.forward(test_input)
                    if torch.isfinite(out[0]).all():
                        return arch
                except Exception as e:
                    pass
                iters += 1
        else:
            raise Exception(f"Unknown generation type {self.generation_type}")

    def generate_dense(self, n_nodes=16)->Architecture:
        self.architecture = Architecture()
        # Always start with an input node
        self.architecture.add_node(0, Input())
        stopflag = False
        rooting_p = 1
        AvailableModules = self.available_modules
        while not stopflag:            
            # Add a new node
            module_type = random.choice(AvailableModules)
            if module_type == Normalizer:
                print(f"Normalizer added")
            module_parameters = module_type.random_parameters()
            lastnode_id = self.architecture.append_node(module_type(*module_parameters))
        
            linked=False
            if not module_type._mapping_type == MappingType.SOURCE:
                for node in list(self.architecture.nodes)[:-1]:
                    if random.random()<rooting_p:
                        self.architecture.add_edge(node, lastnode_id)
                        rooting_p = rooting_p * 0.96
                        linked=True
                if not linked:
                    #self.architecture.remove_node(lastnode_id)
                    stopflag = True

        # remove all the nodes that can't reach the output
        # we define the output as the node with the most ancestors
        outputnode = 0
        outputancestors = 0
        for node in self.architecture.nodes:
            ancestors = networkx.ancestors(self.architecture, node)
            if len(ancestors) > outputancestors and 0 in ancestors :
                outputnode = node
                outputancestors = len(ancestors)

        #check if every node is reachable from the output
        nodesToRemove = []
        for node in self.architecture.nodes:
            if node not in networkx.ancestors(self.architecture, outputnode) and not node == outputnode:
                nodesToRemove.append(node)
        for node in nodesToRemove:
            self.architecture.remove_node(node)
        #check if the architecture is still valid
        if not self.architecture.isValid():
            raise Exception("The architecture is not valid after the generation")
        return self.architecture

    def generate_order_agnostic(self,n_nodes=16)->Architecture:
        """
        first generate n_nodes nodes, then connect them
        """
        self.architecture = Architecture()
        # Always start with an input node
        self.architecture.add_node(0, Input())
        
        rooting_p = random.choice([0.25, 0.3, 0.35, 0.40, 0.45, 0.5, 0.55, 0.6])
        AvailableModules = self.available_modules
        for i in range(n_nodes):
            # Add a new node
            module_type = random.choice(AvailableModules)
            module_parameters = module_type.random_parameters()
            lastnode_id = self.architecture.append_node(module_type(*module_parameters))
        
        for source_node in list(self.architecture.nodes):
            for target_node in list(self.architecture.nodes):
                if target_node != source_node and self.architecture.nodes[target_node]['module'].mapping_type != MappingType.SOURCE:
                    if random.random()<rooting_p and target_node not in list(networkx.ancestors(self.architecture, source_node)):
                        self.architecture.add_edge(source_node, target_node)
        
        # Get all non-source nodes with no source in ancestors
        problematicNodes = []
        for node in list(self.architecture.nodes): # Update the list of problematic nodes
                if self.architecture.nodes[node]['module'].mapping_type != MappingType.SOURCE:
                    hasSourceInAncestors = False
                    for ancestor in networkx.ancestors(self.architecture, node):
                        if self.architecture.nodes[ancestor]['module'].mapping_type == MappingType.SOURCE:
                            hasSourceInAncestors = True
                            break
                    if not hasSourceInAncestors:
                        problematicNodes.append(node)
        baseMaxAttempts = 2000
        maxAttempts = baseMaxAttempts
        while len(problematicNodes) > 0 and maxAttempts > 0:

            for problematicNode in problematicNodes:
                for target_node in list(self.architecture.nodes):
                    if target_node != problematicNode and self.architecture.nodes[target_node]['module'].mapping_type != MappingType.SOURCE:
                        rdm = random.random()<rooting_p
                        if random.random()<rooting_p and not problematicNode in list(networkx.ancestors(self.architecture, target_node)):
                            self.architecture.add_edge(target_node, problematicNode)
                    
            
            maxAttempts -= 1
            problematicNodes = []
            for node in list(self.architecture.nodes): # Update the list of problematic nodes
                if self.architecture.nodes[node]['module'].mapping_type != MappingType.SOURCE:
                    hasSourceInAncestors = False
                    for ancestor in networkx.ancestors(self.architecture, node):
                        if self.architecture.nodes[ancestor]['module'].mapping_type == MappingType.SOURCE:
                            hasSourceInAncestors = True
                            break
                    if not hasSourceInAncestors:
                        problematicNodes.append(node)

        if maxAttempts == 0:
            raise Exception(f"Could not generate a valid architecture after {baseMaxAttempts} attempts")
                  
                        
        # cleaning

        # remove all the nodes that can't reach the output
        # we define the output as the non-source node linked to input with the most ancestors
        outputnode = 0
        outputancestors = 0
        for node in self.architecture.nodes:
            ancestors = networkx.ancestors(self.architecture, node)
            if len(ancestors) > outputancestors and 0 in ancestors and self.architecture.nodes[node]['module'].mapping_type != MappingType.SOURCE:
                outputnode = node
                outputancestors = len(ancestors)

        #check if every node is reachable from the output
        nodesToRemove = []
        for node in self.architecture.nodes:
            if node not in networkx.ancestors(self.architecture, outputnode) and not node == outputnode:
                nodesToRemove.append(node)
        for node in nodesToRemove:
            self.architecture.remove_node(node)
        #check if the architecture is still valid
        if not self.architecture.isValid():
            raise Exception("The architecture is not valid after the generation")
        return self.architecture

        

