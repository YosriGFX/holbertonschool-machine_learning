#!/usr/bin/env python3
'''Deep Neural Network'''
import numpy as np
import matplotlib.pyplot as plt
import pickle


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
        A = self.forward_prop(X)[0]
        return (
            np.where(A >= 0.5, 1, 0),
            self.cost(Y, A)
        )

    def gradient_descent(self, Y, cache, alpha=0.05):
        '''Calculates one pass of gradient
        descent on the deep neural network'''

        weight_copy = self.weights.copy()
        dzi = np.subtract(
            self.cache['A' + str(self.L)], Y
        )
        for i in reversed(range(1, self.L + 1)):
            b = weight_copy['b' + str(i)]
            if i == self.L:
                np.subtract(
                    self.cache['A' + str(self.L)], Y
                )
            else:
                w = weight_copy['W' + str(i + 1)]
                dzi = np.multiply(
                    (
                        self.cache[
                            'A' + str(i)
                        ] * (
                            1 - self.cache['A' + str(i)]
                        )
                    ), np.matmul(w.T, dzi)
                )
            self.__weights['b' + str(i)] = weight_copy[
                'b' + str(i)
            ] - alpha * np.mean(
                    dzi, axis=1, keepdims=True
            )
            self.__weights['W' + str(i)] = weight_copy[
                'W' + str(i)
            ] - alpha * np.matmul(
                    dzi, self.cache['A' + str(i - 1)].T
            ) / Y.shape[1]

    def train(
        self, X, Y,
        iterations=5000,
        alpha=0.05,
        verbose=True,
        graph=True,
        step=100
    ):
        '''Trains the neuron'''
        if type(iterations) != int:
            raise TypeError('iterations must be an integer')
        if iterations < 0:
            raise ValueError('iterations must be a positive integer')
        if type(alpha) != float:
            raise TypeError('alpha must be a float')
        if alpha < 0:
            raise ValueError('alpha must be positive')
        if verbose or graph:
            if type(step) != int:
                raise TypeError('step must be an integer')
            if (step < 0) or (step > iterations):
                raise ValueError('step must be positive and <= iterations')
        allCost, stepper = [], 0
        for i in range(iterations):
            self.gradient_descent(Y, self.forward_prop(X)[1], alpha)
            allCost.append(self.cost(Y, self.cache['A' + str(self.L)]))
            if verbose and (i - 1 == stepper - 1):
                print(
                    'Cost after {} iterations: {}'.format(
                        i, allCost[i]
                    )
                )
                stepper += step
        evaluation, cost = self.evaluate(X, Y)
        i += 1
        print("Cost after {} iterations: {}".format(i, cost))
        if graph:
            plt.plot(allCost)
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()
        return (evaluation, cost)

    def save(self, filename):
        '''Saves the instance object
        to a file in pickle format'''
        if type(filename) is str:
            if filename[-4:] != '.pkl':
                filename = filename[:] + '.pkl'
            file = open(filename, 'wb')
            pickle.dump(self, file)
        else:
            return

    @staticmethod
    def load(filename):
        '''Loads a pickled
        DeepNeuralNetwork object'''
        try:
            file = open(filename, 'rb')
            return pickle.load(file)
        except Exception:
            return None
