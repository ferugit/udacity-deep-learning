import numpy as np
import math

# Write a function that takes as input a list of numbers, and returns
# the list of values given by the softmax function.
def softmax(L):
    softmax_values = []
    denominator = 0
    for i in L:
        denominator += math.exp(i)
    for i in L:
        softmax_values.append(math.exp(i)/denominator)
    return softmax_values


"""
def softmax(L):
    expL = np.exp(L)
    sumExpL = sum(expL)
    result = []
    for i in expL:
        result.append(i*1.0/sumExpL)
    return result
    
"""

if __name__ == "__main__":
    L = [-1.0, 2.0, 0.5, 1.2]
    probs = softmax(L)
    print(probs)