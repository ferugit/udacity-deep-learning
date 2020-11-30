import numpy as np
from softmax_function import softmax

# Write a function that takes as input two lists Y, P,
# and returns the float corresponding to their cross-entropy.
def cross_entropy(Y, P):
    CE = 0
    for i in range(len(Y)):
        CE += (Y[i]*np.log(P[i]) +(1-Y[i])*np.log(1-P[i]))*(-1)
    return CE

"""
def cross_entropy(Y, P):
    Y = np.float_(Y)
    P = np.float_(P)
    return -np.sum(Y * np.log(P) + (1 - Y) * np.log(1 - P))
"""


if __name__ == "__main__":
    out = [-1.0, 7.0, 0.01, 0.2]
    labels = [0, 1, 0, 0]
    probs = softmax(out)
    print(probs)
    print(cross_entropy(labels, probs))
    