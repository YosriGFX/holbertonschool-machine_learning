#!/usr/bin/env python3
'''Neural Network'''
import numpy as np


class NeuralNetwork():
    '''Neural Network defines a neural
    network with one hidden layer
    performing binary classification'''

    def __init__(self, nx, nodes):
        '''Class Constructor'''
        if type(nx) != int:
            raise TypeError("nx must be an integer")
        elif nx < 1:
            raise ValueError("nx must be a positive integer")
        elif type(nodes) != int:
            raise TypeError("nodes must be an integer")
        elif nodes < 1:
            raise ValueError("nodes must be a positive integer")
        else:
            self.__W1 = np.random.normal(size=(nodes, nx))
            self.__b1 = np.array([[np.array(0.)]] * nodes)
            self.__A1 = 0
            self.__W2 = np.random.normal(size=(1, nodes))
            self.__b2 = 0
            self.__A2 = 0

    @property
    def W1(self):
        '''W'''
        return self.__W1

    @property
    def b1(self):
        '''W'''
        return self.__b1

    @property
    def A1(self):
        '''W'''
        return self.__A1

    @property
    def W2(self):
        '''W'''
        return self.__W2

    @property
    def b2(self):
        '''W'''
        return self.__b2

    @property
    def A2(self):
        '''W'''
        return self.__A2

    def forward_prop(self, X):
        '''Calculates the forward
        propagation of the neural network'''
        self.__A1 = self.sigmoid(
            np.matmul(self.__W1, X) + self.__b1
        )
        self.__A2 = self.sigmoid(
            np.matmul(self.__W2, self.__A1) + self.__b2
        )
        return (self.__A1, self.__A2)

    def sigmoid(self, X):
        '''Sigmoid function'''
        return 1 / (1 + np.exp(-X))

    def cost(self, Y, A):
        '''Calculates the cost of the
        model using logistic regression'''
        m = A.shape[1]
        cost = (
                -(1 / m)
            ) * (np.sum(
                    (
                        Y * np.log(A)
                    ) + ((
                            1 - Y
                        ) * np.log(
                            1.0000001 - A
                        )
                    )
                )
            )
        return cost

    def evaluate(self, X, Y):
        '''Evaluates the neuron’s predictions'''
        A = self.forward_prop(X)[1]
        return (
            np.where(A >= 0.5, 1, 0),
            self.cost(Y, A)
        )

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        '''Calculates one pass of gradient
        descent on the neuron'''
        self.__W1 = np.add(
            self.W1,
            -alpha * np.matmul(
                (
                    np.dot(
                        self.__W2.T,
                        (A2 - Y)
                    ) * A1 * (1 - A1)
                ),
                X.T
            ) / X.shape[1]
        )
        self.__b1 += np.mean(
            (
                np.dot(
                    self.__W2.T,
                    (A2 - Y)
                ) * A1 * (1 - A1)
            ),
            axis=1,
            keepdims=True
        ) * -alpha
        self.__W2 = np.add(
            self.W2,
            -alpha * np.matmul(
                A2 - Y,
                self.A1.T
            ) / self.A1.shape[1]
        )
        self.__b2 += np.mean(
            (A2 - Y),
            axis=1,
            keepdims=True
        ) * -alpha
