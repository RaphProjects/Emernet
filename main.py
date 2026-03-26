import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

from modules.base import ModuleType
from modules.operations import *
from modules.learnable import *
from modules.input import *
from graph.architecture import *
from graph.executor import *
from graph.generator import *
from tournament import arena
from tournament.arena import *
from scipy.stats import pearsonr

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

def patch_module_types(arch):
    for nid in arch.nodes:
        mod = arch.nodes[nid]["module"]

        if isinstance(mod, Input):
            mod.module_type = ModuleType.INPUT
        elif isinstance(mod, LearnableParameter):
            mod.module_type = ModuleType.LEARNABLE
        elif isinstance(mod, Activation):
            mod.module_type = ModuleType.ACTIVATION
        elif isinstance(mod, (Add, MatMul, Mult)):
            mod.module_type = ModuleType.BASIC
        elif isinstance(mod, (Pooling, Concat, Transpose, Shift, Split)):
            mod.module_type = ModuleType.STRUCTURAL
        elif isinstance(mod, Normalizer):
            mod.module_type = ModuleType.NORM

    return arch

    
#twolayersMLP()

arena = Arena(n_fights=48, architecture_size=12, arena_contestants=3, dataset_size=512, train_test_split=0.7, generation_type="agnostic", verbose=False, report=False)
mlp = arena.make_mlp([32,16,16])
pareto_best = Architecture.load("pareto_best.pkl")
pareto_best = patch_module_types(pareto_best)
mlp_sizes = [[16,32,32]]
archs = []
for size in mlp_sizes:
    archs.append(arena.make_mlp(size))


#archs.append(pareto_best)

wrs, occam_scores, norm_learn, norm_simp = arena.occam_test(archs, n_archs=9, verbose=True, randomizeHP=True)
print(f"Winrates : {wrs} \n Occam scores : {occam_scores} \n Learnabilities : {norm_learn} \n Simplicities : {norm_simp}")
'''
print("nodes:", list(pareto_best.nodes))
print("edges:", list(pareto_best.edges))
print("MLP params:", mlp.parameter_count(), "| Pareto params:", pareto_best.parameter_count())



mlp_wrs = []
pareto_winner_wrs = []
for i in range(10):
    wrs, occam_scores, learnabilities, simplicities = arena.occam_test([mlp,pareto_best], n_archs=18, verbose=True, randomizeHP=True)
    mlp_wrs.append(wrs[0])
    pareto_winner_wrs.append(wrs[1])

avg_mlp_wr = sum(mlp_wrs)/len(mlp_wrs)
avg_pareto_winner_wr = sum(pareto_winner_wrs)/len(pareto_winner_wrs)




simp_bals, avg_simp_bal, std_simp_bal = arena.tune_simp_bal(n_archs=12, n_rounds=4, verbose=True, randomizeHP=True, use_MLPs=True)
print(f"simp_bals: {simp_bals}, avg: {avg_simp_bal}, std: {std_simp_bal}")

winner, occam_scores, max_score_idx, learnability_scores, simplicity_scores = arena.pareto_selection(n_archs=18, n_rounds=5, verbose=True, randomizeHP=True)
print(f"Winner score: {occam_scores[max_score_idx]}, scores = {occam_scores}")
winner.describe()
winner.save("pareto_winner_16archs_3rounds.pkl")


generator = Generator(generation_type="agnostic")
architectures = [generator.generate(n_nodes=12) for _ in range(1)]
architectures.append(Architecture.load("O_winner_2archs.pkl"))
architectures.append(Architecture.load("O_winner_3archs.pkl"))

architectures.append(Architecture.load("O_winner_4archs.pkl"))

architectures.append(Architecture.load("O_winner_20archs.pkl"))
architectures.append(Architecture.load("O_winner_23archs.pkl"))

'''
'''
architectures.append(Architecture.load("O_winner_24archs.pkl"))
print(arena.test_real_correlation(architectures=architectures, n_archs_test=5, simp_bal=0.3, verbose = True, real_iter = 60))



mlp = arena.make_mlp([32,16])
winner = Architecture.load("winner_24_opponnents_12_nodes_91wr.pkl")

mlp_scores, mlp_contestant_scores = arena.get_distinction(mlp, verbose=True)
winner_scores, winner_contestant_scores = arena.get_distinction(winner, verbose=True)

print(f"MLP avg score: {sum(mlp_scores)/len(mlp_scores)}, Winner avg score: {sum(winner_scores)/len(winner_scores)}")
print(f"MLP avg contestant score: {sum(mlp_contestant_scores)/len(mlp_contestant_scores)}, Winner avg contestant score: {sum(winner_contestant_scores)/len(winner_contestant_scores)}")


'''
'''
winner, occam_scores, winner_idx, learnabilities, simplicites = arena.occam_selection(n_archs=4, verbose=True, randomizeHP=True, simp_bal=0.3)
print(f"Occam scores : {occam_scores} \n Learnabilities : {learnabilities} \n Simplicities : {simplicites} \n Occam avg score: {sum(occam_scores)/len(occam_scores)}, Winner score: {occam_scores[winner_idx]}")
winner.save("O_winner_4archs.pkl")



winner = Architecture.load("O_winner_23archs.pkl")

mlp = arena.make_mlp([32,16,8])

wrs, occam_scores, learnabilities, simplicities = arena.occam_test([winner,mlp], n_archs=18, verbose=True, randomizeHP=True, simp_bal=0.3)
print(f"Winrates : {wrs} \n Occam scores : {occam_scores} \n Learnabilities : {learnabilities} \n Simplicities : {simplicities} \n Occam avg score: {sum(occam_scores)/len(occam_scores)}")




best_arch, occam_scores, max_score_idx, learnability_scores, simplicity_scores = arena.pareto_selection(n_archs=18, n_rounds=5, verbose=True, randomizeHP=True)
best_arch.save("pareto_best.pkl")

'''
# NOTE - O_winner_2archs might be OP for no reason
# TODO - find the % of random archs beating MLP
# TODO - uniform the input tensors
# TODO - GNN encoding of archs (graph variational autoencoder)
# TODO - A better arch descriptor
