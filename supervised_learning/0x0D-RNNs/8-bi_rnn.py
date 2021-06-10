#!/usr/bin/env python3
'''8. Bidirectional RNN'''
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    '''performs forward propagation for a bidirectional RNN'''
    t = X.shape[0]
    h_0_i = np.zeros(
        (
            t,
            X.shape[1],
            h_0.shape[1]
            )
    )
    h_t_i = np.zeros(
        (
            t,
            X.shape[1],
            h_t.shape[1]
        )
    )
    for a in range(0, t):
        h_0_i[a] = bi_cell.forward(
            h_0, X[a]
        )
        h_0 = h_0_i[a]
    for a in range(0, t)[::-1]:
        h_t_i[a] = bi_cell.backward(
            h_t, X[a]
        )
        h_t = h_t_i[a]
    H = np.concatenate(
        (h_0_i, h_t_i),
        axis=2
    )
    Y = bi_cell.output(H)
    return H, Y
