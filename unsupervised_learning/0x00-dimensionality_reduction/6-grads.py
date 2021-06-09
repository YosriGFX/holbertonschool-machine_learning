#!/usr/bin/env python3
'''6. Gradients'''
import numpy as np
Q_affinities = __import__('5-Q_affinities').Q_affinities


def grads(Y, P):
    '''calculates the gradients of Y'''
    a, n_dim = Y.shape
    Q, num = Q_affinities(Y)
    dY = np.zeros(
        (a, n_dim)
    )
    b = np.expand_dims(
        ((P - Q) * num).T,
        axis=2
    )
    for i in range(a):
        diff = Y[i, :] - Y
        dY[i, :] = np.sum(
            (b[i, :] * diff), 0
        )
    return dY, Q
