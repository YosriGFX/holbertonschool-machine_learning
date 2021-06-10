#!/usr/bin/env python3
'''2. Update Gaussian Process'''
import numpy as np


class GaussianProcess:
    '''represents a noiseless 1D Gaussian process'''
    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        '''Class constructor'''
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        '''calculates the covariance kernel
        matrix between two matrices'''
        return self.sigma_f ** 2 * np.exp(
            -0.5 / self.l ** 2 * (
                np.sum(
                    X1 ** 2, 1
                ).reshape(
                    -1, 1
                ) + np.sum(
                    X2 ** 2, 1
                ) - 2 * np.dot(
                    X1, X2.T
                )
            )
        )

    def predict(self, X_s):
        '''predicts the mean and standard
        deviation of points in a Gaussian process'''
        K = self.K
        Inv = np.linalg.inv(K)
        a = self.kernel(self.X, X_s)
        mu = a.T.dot(Inv).dot(self.Y)
        mu = mu.reshape(-1)
        sigma = self.kernel(X_s, X_s) - a.T.dot(Inv).dot(a)
        sigma = np.diag(sigma)
        return mu, sigma

    def update(self, X_new, Y_new):
        '''updates a Gaussian Process'''
        self.X = np.append(
            self.X, X_new
        )[:, np.newaxis]
        self.Y = np.append(
            self.Y, Y_new
        )[:, np.newaxis]
        self.K = self.kernel(
            self.X, self.X
        )
