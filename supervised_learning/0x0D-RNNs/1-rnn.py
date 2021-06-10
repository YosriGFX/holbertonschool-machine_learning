#!/usr/bin/env python3
'''1. RNN'''
import numpy as np


def rnn(rnn_cell, X, h_0):
    '''performs forward propagation for a simple RNN'''
    T = X.shape[0]
    m = X.shape[1]
    a, b = rnn_cell.Wy.shape
    a = np.zeros((T + 1, m, a))
    Y = np.zeros((T, m, b))
    for t in range(1, T + 1):
        a[t], Y[t - 1] = rnn_cell.forward(
            a[t - 1],
            X[t - 1]
        )
    return a, Y
