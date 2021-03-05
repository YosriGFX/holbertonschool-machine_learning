#!/usr/bin/env python3
'''Neuron'''
import numpy as np


class Neuron:
    '''Neuron defines a single neuron
    performing binary classification'''

    def __init__(self, nx):
        '''Class Constructor'''
        if type(nx) != int:
            raise TypeError('TypeError')
        elif nx < 1:
            raise ValueError('nx must be a positive integer')
        else:
            self.nx = nx
            self.W = np.random.normal(size=(1, self.nx))
            self.b = 0
            self.A = 0
