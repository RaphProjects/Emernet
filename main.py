import torch
import numpy as np
from modules.base import ModuleType
from modules.operations import *
from modules.learnable import *
from modules.input import *
from graph.architecture import *
from graph.executor import *
from graph.generator import *
from tournament import arena
from tournament.arena import *

def linReg():
    architecture = Architecture()

    inputTens = torch.randn(2,3,4)
    outputTargetTens = torch.randn(2,1,1)
    inputModule = Input()
    inputModule.set_data(inputTens)
    weights = LearnableParameter((4,8))
    bias = LearnableParameter((1,1,8))
    matmul = MatMul()
    addition = Add()

    architecture.add_node(0,inputModule)
    architecture.add_node(1,weights)
    architecture.add_node(2,bias)
    architecture.add_node(3,matmul)
    architecture.add_node(4,addition)

    architecture.add_edge(0,3)
    architecture.add_edge(1,3)
    architecture.add_edge(3,4)
    architecture.add_edge(2,4)

    print(architecture.isValid())
    print(list(networkx.topological_sort(architecture)))
    print(list(architecture.nodes))

    executor = Executor(architecture)
    executor.set_Output_Adapter(inputTens, outputTargetTens.shape)
    output = executor.forward(inputTens)
    optimizer = torch.optim.Adam(executor.parameters(), lr=0.01)
    for i in range(101):
        output = executor.forward(inputTens)
        loss = torch.nn.functional.mse_loss(output[0], outputTargetTens)
        if i % 10 == 0:
            print(f"Loss: {loss.item()}")
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

def twolayersMLP():
    architecture = Architecture()

    inputTens = torch.randn(16,3,4)
    outputTargetTens = torch.randn(16,1,4)
    inputModule = Input()
    inputModule.set_data(inputTens)
    architecture.add_node(0,inputModule)

    architecture.append_node(LearnableParameter((1,4,13)))
    architecture.append_node(LearnableParameter((1,1,7)))
    architecture.append_node(MatMul())
    architecture.append_node(Add())
    architecture.append_node(Activation())

    architecture.append_node(LearnableParameter((1,13,4)))
    architecture.append_node(LearnableParameter((1,1,4)))
    architecture.append_node(MatMul())
    architecture.append_node(Add())

    #First layer connections
    architecture.add_edge(0,3)
    architecture.add_edge(1,3)
    architecture.add_edge(3,4)
    architecture.add_edge(2,4)
    architecture.add_edge(4,5)
    #Second layer connections
    architecture.add_edge(5,8)
    architecture.add_edge(6,8)
    architecture.add_edge(7,9)
    architecture.add_edge(8,9)

    print(architecture.isValid())
    print(list(networkx.topological_sort(architecture)))

    executor = Executor(architecture)
    executor.fit(inputTens, outputTargetTens, verbose=True, lr=0.002, max_iter=1000, batch_size=16, patience = 10, min_delta = 1e-7)

# twolayersMLP()

'''
myGenerator = Generator()

architecture = myGenerator.generate_order_agnostic(n_nodes=8)


architecture = myGenerator.generate(n_nodes=8)
#print(architecture.isValid())
#print(list(networkx.topological_sort(architecture)))
architecture.describe()
inputTens = torch.randn(64,2,32)
outputTargetTens = inputTens+torch.randn(64,2,32)*0.1

executor = Executor(architecture)
executor.fit(inputTens, outputTargetTens, verbose=True, lr=0.002, max_iter=20, batch_size=8, patience = 32, min_delta = 1e-7, cpu = False)
#executor.fit(inputTens, outputTargetTens, verbose=True, lr=0.01, max_iter=100, batch_size=8, patience = 10, min_delta = 1e-7, cpu = True)
'''


arena1 = Arena(n_fights=5, architecture_size=12, arena_contestants=3, dataset_size=512, train_test_split=0.7, generation_type="agnostic", verbose=False, report=False)
winner_scores, winners = arena1.start()

WinArch1 = winners[-1]
arch_scores, n_wins1 = arena1.test(WinArch1,n_test=6)
mlp_scores1, mlpwinrate1 = arena1.test_mlp(WinArch1,mlp_n_tests=10, mlp_hidden_sizes=[32,32,16])


arena2 = Arena(n_fights=8, architecture_size=12, arena_contestants=3, dataset_size=512, train_test_split=0.7, generation_type="agnostic", verbose=False, report=False)
winrate2, winners = arena2.start()


WinArch2 = winners[-1]
arch_scores, n_wins2 = arena2.test(WinArch2,n_test=6)
mlp_scores, mlpwinrate2 = arena2.test_mlp(WinArch2,mlp_n_tests=10, mlp_hidden_sizes=[32,32,16])

print(f"Winner 1 winrate: {n_wins1/6}")
print(f"Winner 2 winrate: {n_wins2/6}")
print(f"Winner 1 MLP winrate: {mlpwinrate1}")
print(f"Winner 2 MLP winrate: {mlpwinrate2}")

print(f"Winner 1 MLP score: {mlp_scores1}")

