#!/usr/bin/env python3
'''Momentum'''
import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    '''Function that updates a variable
    using the gradient descent with
    momentum optimization algorithm'''
    vt = np.add(
        v * beta1,
        grad * (1 - beta1)
    )
    return np.subtract(
        var,
        np.multiply(vt, alpha)
    ), vt
