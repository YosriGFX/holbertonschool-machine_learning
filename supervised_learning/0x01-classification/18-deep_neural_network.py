#!/usr/bin/env python3
'''Deep Neural Network'''
import numpy as np


class DeepNeuralNetwork:
    '''Deep Neural Network defines
    deep neural network performing
    binary classification'''

    def __init__(self, nx, layers):
        '''Class Constructor'''
        if type(nx) != int:
            raise TypeError('nx must be an integer')
        elif nx < 1:
            raise ValueError('nx must be a positive integer')
        elif (
            type(layers) != list
        ) or (
            len(layers) < 1
        ) or (
            min(layers) < 1
        ):
            raise TypeError("layers must be a list of positive integers")
        else:
            self.__L = len(layers)
            self.__cache = {}
            self.__weights = {
                "W1": np.random.randn(
                    layers[0], nx
                ) * np.sqrt(2 / nx),
                "b1": np.zeros((layers[0], 1))
            }
            for i in range(1, self.L):
                self.weights[
                    'W{}'.format(i + 1)
                ] = np.random.randn(
                        layers[i],
                        layers[i - 1]
                    ) * np.sqrt(2 / layers[i - 1])
                self.weights[
                    'b{}'.format(i + 1)
                ] = np.zeros((layers[i], 1))

    @property
    def L(self):
        '''L'''
        return self.__L

    @property
    def cache(self):
        '''cache'''
        return self.__cache

    @property
    def weights(self):
        '''weights'''
        return self.__weights

    def forward_prop(self, X):
        '''Calculates the forward'''
        self.__cache['A0'] = X
        for i in range(1, self.L + 1):
            self.__cache[
                'A' + str(i)
            ] = self.sigmoid(
                    self.__cache['A'+str(i - 1)],
                    self.weights['W'+str(i)],
                    self.weights['b'+str(i)]
                )
        return self.cache['A' + str(self.L)], self.cache

    def sigmoid(self, X=None, w=None, b=None, x=None):
        '''Sigmoid function'''
        if x:
            return 1 / (1 + np.exp(-x))
        else:
            return 1 / (1 + np.exp(
                -np.add(
                    np.matmul(w, X), b
                )
            ))
