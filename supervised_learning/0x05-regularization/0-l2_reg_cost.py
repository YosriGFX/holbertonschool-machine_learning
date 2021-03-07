#!/usr/bin/env python3
''' L2 Regularization Cost'''
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    '''Function that calculates
    the cost of a neural network
    with L2 regularization'''
    tmp = 0
    for i in range(1, L + 1):
        tmp += np.linalg.norm(
            weights['W' + str(i)],
            ord='fro'
        ) ** 2
    return (
        cost + (
            lambtha * tmp
        ) / (
         2 * m
        )
    )
