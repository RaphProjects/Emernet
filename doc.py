'''IN/OUT dims :

    Weight : x input vectors -> x output vectors (independantly weighted)
    Add : x input vectors -> 1 output vector
    Activation : x input vectors -> x output vectors (all independantly activated)
    Aggregation : x input vectors -> x output scalars (all independantly aggregated)
    Memory state : x input vectors -> x output vectors (the ones inputed at the previous step)
    Memory tape : x input vectors -> T*x output vectors (with T the position of the element)
    Concatenate : x input vectors -> 1 output vector

MLP avg score: 70.47845018121056, Winner avg score: 2.3837992025481864
T(n)=(5/24)n(n-1) 
n = (24*T(n))/5

# TODO - research metaheuristique + theorie de l'apprentissage statistique + VC dimension
20 - 0.955191577769701
23 - 1.3011372052075836 - Winrates : [0.9333333333333333, 0.6666666666666666]
against mlp :
average learnability score: -2.790642113527907e-16
average simplicity score: -9.412760426169805e-17
'''
import math

def n_archs_from_minutes(t):
    return (1 + math.sqrt(1 + (96/5)*t)) / 2

print(n_archs_from_minutes(110))
    










"""
382.2779071331024
[0.7469304076094267, 0.4510897695167949, 0.12143144596785022, 0.46276275266871164]
A QD method like MAP-Elites could keep an archive over behavioral descriptors such as:

parameter count
graph depth
graph density
ratio of Add/MatMul/Activation
learnability
simplicity
stability class
"""
