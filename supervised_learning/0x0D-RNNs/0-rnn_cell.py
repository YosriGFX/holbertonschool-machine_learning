#!/usr/bin/env python3
'''0. RNN Cell'''
import numpy as np


class RNNCell:
    '''represents a cell of a simple RNN'''
    def __init__(self, i, h, o):
        '''Class constructor'''
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def softmax(h):
        '''softmax activation function'''
        Exp = np.exp(h)
        Exp = Exp / np.sum(
            Exp, 1, keepdims=True
        )
        return Exp

    def forward(self, h_prev, x_t):
        '''represent the weights and biases of the cell'''
        h_next = np.tanh(
            np.hstack(
                (h_prev, x_t)
            ) @ self.Wh + self.bh
        )
        y = self.softmax(
            h_next @ self.Wy + self.by
        )
        return h_next, y
