#!/usr/bin/env python3
'''Mean and Covariance'''
import numpy as np


def mean_cov(X):
    '''Calculates the mean and covariance of a data set'''
    if type(X) is np.ndarray and len(X.shape) == 2:
        if X.shape[0] > 2:
            a, b = X.shape
            Mean = np.mean(
                X, axis=0
            ).reshape((1, b))
            Ones = np.ones(
                (a, a)
            )
            score = X - np.matmul(Ones, X) * (1 / a)
            return Mean, np.matmul(
                    score.T, score
                ) / (a - 1)
        else:
            raise ValueError("X must contain multiple data points")
    else:
        raise TypeError("X must be a 2D numpy.ndarray")
