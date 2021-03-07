#!/usr/bin/env python3
'''Gradient Descent with L2 Regularization'''
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    '''Function that updates the
    weights and biases of a neural
    network using gradient descent
    with L2 regularization:'''
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
                1 - (a ** 2),
                np.matmul(w.T, dzi)
            )
        dwi = np.matmul(dzi, a1.T) / Y.shape[1]
        dbi = np.mean(
            dzi, axis=1, keepdims=True
        )
        weights['b' + str(t)] = weightBias[
            'b' + str(t)
        ] - alpha * (dbi)
        dwi = dwi + lambtha / Y.shape[
            1
        ] * weightBias['W' + str(t)]
        weights['W' + str(t)] = weightBias[
            'W'+str(t)
        ] - alpha * (dwi)
