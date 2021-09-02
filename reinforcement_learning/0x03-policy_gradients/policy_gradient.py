#!/usr/bin/env python3
'''0. Simple Policy function'''
import numpy as np


def soft_max(value):
    '''A function that Computes the softmax'''
    exp = np.exp(
        value - np.max(value)
    )
    return exp / np.sum(exp)


def policy(matrix, weight):
    '''A function that computes to policy
    with a weight of a matrix'''
    return soft_max(matrix @ weight)
