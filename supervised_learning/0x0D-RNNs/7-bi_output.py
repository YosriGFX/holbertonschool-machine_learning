#!/usr/bin/env python3
'''7. Bidirectional Output'''
import numpy as np


class BidirectionalCell:
    '''represents a bidirectional cell of an RNN'''
    def __init__(self, i, h, o):
        '''Class constructor '''
        self.Whf = np.random.randn(i + h, h)
        self.bhf = np.zeros((1, h))
        self.Whb = np.random.randn(i + h, h)
        self.bhb = np.zeros((1, h))
        self.Wy = np.random.randn(2 * h, o)
        self.by = np.zeros((1, o))

    @staticmethod
    def softmax(h):
        '''softmax activation function'''
        Exp = np.exp(h)
        Exp = Exp / np.sum(
            Exp, 1, keepdims=True
        )
        return Exp

    def forward(self, h_prev, x_t):
        '''calculates the hidden state in the
        forward direction for one time step'''
        h_next = np.tanh(
            np.hstack(
                (h_prev, x_t)
            ) @ self.Whf + self.bhf
        )
        return h_next

    def backward(self, h_next, x_t):
        '''calculates the hidden state in the
        backward direction for one time step'''
        h_prev = np.tanh(
            np.hstack(
                (h_next, x_t)
            ) @ self.Whb + self.bhb
        )
        return h_prev

    def output(self, H):
        '''calculates all outputs for the RNN'''
        Y = self.softmax(
            H @ self.Wy + self.by
        )
        return Y
