import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

def test_real_correlation(architectures,n_test=16,simp_bal=0.3, verbose = True):
    # We evaluate every architecture on both the arena and the real data set
    # We compute the correlation between the architectures' scores on the real data set and the scores on the arena

    ############################ Arena metrics ############################
    learnabilities = []
    simplicities = []
    occam_scores = []
    for arch in architectures: 
        wrs, occam_scores_round, learnabilities_round, simplicities_round = arena.occam_test([arch], n_archs=n_test, verbose=False, randomizeHP=True, simp_bal=simp_bal)
        learnabilities.append(learnabilities_round[0])
        simplicities.append(simplicities_round[0])
        occam_scores.append(occam_scores_round[0])
    arena_metrics = {"learnability": learnabilities, "simplicity": simplicities, "occam_score": occam_scores}
    ############################ Real data set metrics ############################

    real_dataset_metrics = {} # dict of ["dataset-metrics"]->list of values
    for arch in architectures:
        res = arena.realDataSet_test(arch, verbose=False)
        for dataset in res.keys():
            for metric in res[dataset].keys():
                real_dataset_metrics[f"{dataset}-{metric}"].append(res[dataset][metric])

    
    ########################### Compute correlations ##############################

    correlations = {} # dict of ["arenametric-dataset-realdatametric"]->correlation
    for arenametric in arena_metrics:
        for realdatametric in real_dataset_metrics:
            correlations[f"{arenametric}-{realdatametric}"] = pearsonr(arena_metrics[arenametric], real_dataset_metrics[realdatametric])[0]
    if verbose:
        print(correlations)
    return correlations
    
#twolayersMLP()

arena = Arena(n_fights=48, architecture_size=12, arena_contestants=3, dataset_size=512, train_test_split=0.7, generation_type="agnostic", verbose=False, report=False)
'''

mlp = arena.make_mlp([32,16])
winner = Architecture.load("winner_24_opponnents_12_nodes_91wr.pkl")

mlp_scores, mlp_contestant_scores = arena.get_distinction(mlp, verbose=True)
winner_scores, winner_contestant_scores = arena.get_distinction(winner, verbose=True)

print(f"MLP avg score: {sum(mlp_scores)/len(mlp_scores)}, Winner avg score: {sum(winner_scores)/len(winner_scores)}")
print(f"MLP avg contestant score: {sum(mlp_contestant_scores)/len(mlp_contestant_scores)}, Winner avg contestant score: {sum(winner_contestant_scores)/len(winner_contestant_scores)}")


'''

winner, occam_scores, winner_idx, learnabilities, simplicites = arena.occam_selection(n_archs=4, verbose=True, randomizeHP=True, simp_bal=0.3)
print(f"Occam scores : {occam_scores} \n Learnabilities : {learnabilities} \n Simplicities : {simplicites} \n Occam avg score: {sum(occam_scores)/len(occam_scores)}, Winner score: {occam_scores[winner_idx]}")
winner.save("O_winner_4archs.pkl")

'''

winner = Architecture.load("O_winner_23archs.pkl")

mlp = arena.make_mlp([32,16,8])

wrs, occam_scores, learnabilities, simplicities = arena.occam_test([winner,mlp], n_archs=18, verbose=True, randomizeHP=True, simp_bal=0.3)
print(f"Winrates : {wrs} \n Occam scores : {occam_scores} \n Learnabilities : {learnabilities} \n Simplicities : {simplicities} \n Occam avg score: {sum(occam_scores)/len(occam_scores)}")


'''

# TODO - make simplicity and learnability in normalized log space to prevent outlier dominance
# TODO - Find a way to make architectures less dense
# TODO - find the % of random archs beating MLP at similar sizes
# TODO - compute the average learnability and simplicity of the architectures - faster comparisons

