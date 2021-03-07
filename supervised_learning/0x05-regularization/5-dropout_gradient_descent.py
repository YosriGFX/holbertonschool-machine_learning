#!/usr/bin/env python3
'''Gradient Descent with Dropout'''
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    '''Function that calculate one pass
    of gradient descent on the neuron'''
    weightBias = weights.copy()
    dzi = np.subtract(
        cache['A' + str(L)], Y
    )
    for t in reversed(range(1, L + 1)):
        a = cache['A' + str(t)]
        a1 = cache['A' + str(t - 1)]
        b = weightBias['b' + str(t)]
        if t == L:
            dzi = np.subtract(
                cache['A' + str(L)], Y
            )
        else:
            w = weightBias['W' + str(t + 1)]
            dzi = np.multiply(
                np.multiply(
                    1 - (a ** 2), cache["D" + str(t)]
                ) / keep_prob,
                np.matmul(w.T, dzi)
            )
        dwi = np.matmul(dzi, a1.T) / Y.shape[1]
        dbi = np.sum(
            dzi, axis=1, keepdims=True
        ) / Y.shape[1]
        weights['b' + str(t)] = weightBias[
            'b' + str(t)
        ] - alpha * (dbi)
        weights['W' + str(t)] = weightBias[
            'W'+str(t)
        ] - alpha * (dwi)
