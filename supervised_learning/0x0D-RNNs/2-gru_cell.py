#!/usr/bin/env python3
'''2. GRU Cell'''
import numpy as np


class GRUCell:
    '''represents a gated recurrent unit'''
    def __init__(self, i, h, o):
        '''lass constructor'''
        self.Wz = np.random.randn(i + h, h)
        self.bz = np.zeros((1, h))
        self.Wr = np.random.randn(i + h, h)
        self.br = np.zeros((1, h))
        self.Wh = np.random.randn(i + h, h)
        self.bh = np.zeros((1, h))
        self.Wy = np.random.randn(h, o)
        self.by = np.zeros((1, o))

    @staticmethod
    def softmax(h):
        '''softmax activation function'''
        Exp = np.exp(h)
        Exp = Exp / np.sum(
            Exp, 1, keepdims=True
        )
        return Exp

    @staticmethod
    def sigmoid(h):
        '''sigmoid activation function'''
        Sigmoid = 1 / (
            1 + np.exp(-h)
        )
        return Sigmoid

    def forward(self, h_prev, x_t):
        '''performs forward propagation for one time step'''
        Hstack = np.hstack(
            (h_prev, x_t)
        )
        z = self.sigmoid(
            Hstack @ self.Wz + self.bz
        )
        r = self.sigmoid(
            Hstack @ self.Wr + self.br
        )
        Hstack = np.hstack(
            (r * h_prev, x_t)
        )
        h_tild = np.tanh(
            Hstack @ self.Wh + self.bh
        )
        h_next = (
            np.ones_like(z) - z
        ) * h_prev + z * h_tild
        y = self.softmax(
            h_next @ self.Wy + self.by
        )
        return h_next, y
