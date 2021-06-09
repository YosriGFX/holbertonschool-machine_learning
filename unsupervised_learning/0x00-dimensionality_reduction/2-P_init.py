#!/usr/bin/env python3
'''2. Initialize t-SNE'''
import numpy as np


def P_init(X, perplexity):
    '''that initializes all variables required
    to calculate the P affinities in t-SNE'''
    a = X.shape[0]
    Sum = np.sum(
        np.square(X),
        axis=1
    )
    D = np.add(
        np.add(
            -2 * np.matmul(X, X.T),
            Sum
        ).T, Sum
    )
    np.fill_diagonal(D, 0)
    return D,\
        np.zeros((a, a)),\
        np.ones((a, 1)),\
        np.log2(perplexity)
