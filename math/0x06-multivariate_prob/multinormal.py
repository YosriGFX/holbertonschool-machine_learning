#!/usr/bin/env python3
'''Mean and Covariance'''
import numpy as np


class MultiNormal():
    '''Multivariate Normal distribution'''

    def __init__(self, data):
        '''class constructor'''
        if type(data) is np.ndarray and len(data.shape) == 2:
            if data.shape[1] > 2:
                a, b = data.shape
                self.mean = (
                    np.mean(data, axis=1)
                ).reshape(a, 1)
                X = data - self.mean
                self.cov = (
                    (
                        np.matmul(X, X.T)
                    ) / (b - 1)
                )
            else:
                raise ValueError("data must contain multiple data points")
        else:
            raise TypeError("data must be a 2D numpy.ndarray")
