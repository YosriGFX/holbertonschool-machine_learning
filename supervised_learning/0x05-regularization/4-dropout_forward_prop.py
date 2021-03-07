#!/usr/bin/env python3
''' Forward Propagation with Dropout'''
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    '''A function that conducts forward
    propagation using Dropout'''
    temp = {}
    temp["A0"] = X
    for t in range(1, L + 1):
        W = weights["W" + str(t)]
        b = weights["b" + str(t)]
        A = temp['A' + str(t-1)]
        Z = np.dot(W, A) + b
        if t != L:
            temp['A' + str(t)] = np.tanh(Z)
            A = temp['A' + str(t)]
            temp["D" + str(t)] = np.random.rand(
                A.shape[0],
                A.shape[1]
            ) < keep_prob
            temp["D" + str(t)] = np.where(
                temp["D" + str(t)] < keep_prob, 0, 1
            )
            temp['A' + str(t)] = np.multiply(
                A, temp["D" + str(t)]
            ) / keep_prob
        else:
            temp['A' + str(t)] = np.exp(Z) / (np.sum(
                np.exp(Z), axis=0)
            )
    return temp
