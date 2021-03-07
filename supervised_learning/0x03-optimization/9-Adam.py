#!/usr/bin/env python3
'''Adam File'''
import numpy as np


def update_variables_Adam(
    alpha,
    beta1,
    beta2,
    epsilon,
    var,
    grad,
    v,
    s,
    t
):
    '''Function that updates a variable
    in place using the Adam optimization
    algorithm'''
    sTime = np.add(
        s * beta2,
        (grad ** 2) * (1 - beta2)
    )
    sC = sTime / (
        1 - beta2 ** t
    )
    vTime = (
        np.add(
            v * beta1,
            grad * (1 - beta1)
        )
    )
    varPer = vTime / (
        1 - beta1 ** t
    )
    newgrad = np.subtract(
        var,
        np.divide(
            np.multiply(varPer, alpha),
            np.sqrt(sC) + epsilon
        )
    )
    return newgrad, vTime, sTime
