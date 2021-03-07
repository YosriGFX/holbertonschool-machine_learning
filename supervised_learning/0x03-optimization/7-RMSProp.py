#!/usr/bin/env python3
'''RMSProp'''
import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    '''Function that updates a variable
    using the RMSProp optimization algorithm'''
    matter = np.add(
        s * beta2,
        (grad ** 2) * (1 - beta2)
    )
    return np.subtract(
        var,
        np.divide(
            np.multiply(grad, alpha),
            (
                np.sqrt(matter) + epsilon
            )
        )
    ), matter
