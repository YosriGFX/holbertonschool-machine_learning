#!/usr/bin/env python3
'''Batch Normalization'''
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    '''Function that normalizes an
    unactivated output of a neural
    network using batch normalization'''
    µ = Z.mean(axis=0)
    return gamma * np.subtract(
        Z, µ
    ) / (
        np.sqrt(
            (
                np.subtract(Z, µ) ** 2
            ).mean(axis=0) + epsilon
        )
    ) + beta
