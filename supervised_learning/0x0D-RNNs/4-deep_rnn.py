#!/usr/bin/env python3
'''4. Deep RNN'''
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    '''performs forward propagation for a deep RNN'''
    a, m, i = X.shape
    l = h_0.shape[0]
    h = h_0.shape[2]
    H = np.zeros(
        (a + 1, l, m, h)
    )
    h, o = rnn_cells[-1].Wy.shape
    Y = np.zeros(
        (a, m, o)
    )
    for i, cells in enumerate(rnn_cells):
        if i == 0:
            for b in range(1, a + 1):
                H[b, i], _ = cells.forward(
                    H[b - 1, i], X[b - 1]
                )
        else:
            for b in range(1, a + 1):
                H[b, i], Y[b - 1] = cells.forward(
                    H[b - 1, i], H[b, i - 1]
                )
    return H, Y
