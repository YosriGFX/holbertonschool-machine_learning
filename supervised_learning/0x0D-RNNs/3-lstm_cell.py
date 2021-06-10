#!/usr/bin/env python3
'''3. LSTM Cell'''
import numpy as np


class LSTMCell:
    '''represents an LSTM unit'''
    def __init__(self, i, h, o):
        self.Wf = np.random.randn(i + h, h)
        self.bf = np.zeros((1, h))
        self.Wu = np.random.randn(i + h, h)
        self.bu = np.zeros((1, h))
        self.Wc = np.random.randn(i + h, h)
        self.bc = np.zeros((1, h))
        self.Wo = np.random.randn(i + h, h)
        self.bo = np.zeros((1, h))
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

    def forward(self, h_prev, c_prev, x_t):
        '''performs forward propagation for one time step'''
        Hstack = np.hstack(
            (h_prev, x_t)
        )
        f = self.sigmoid(
            Hstack @ self.Wf + self.bf
        )
        u = self.sigmoid(
            Hstack @ self.Wu + self.bu
        )
        c_next = f * c_prev + u * np.tanh(
            Hstack @ self.Wc + self.bc
        )
        o = self.sigmoid(
            Hstack @ self.Wo + self.bo
        )
        h_next = o * np.tanh(
            c_next
        )
        y = self.softmax(
            h_next @ self.Wy + self.by
        )
        return h_next, c_next, y
