#!/usr/bin/env python3
'''6. Bidirectional Cell Backward'''
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
