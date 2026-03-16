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


20 - 0.955191577769701

'''
import math

def n_archs_from_minutes(t):
    return (1 + math.sqrt(1 + (96/5)*t)) / 2

print(n_archs_from_minutes(110))
    
