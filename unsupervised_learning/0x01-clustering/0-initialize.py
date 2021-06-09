#!/usr/bin/env python3
'''0. Initialize K-means'''
import numpy as np


def initialize(X, k):
    '''initializes cluster centroids for K-means'''
    if isinstance(X, np.ndarray) and len(X.shape) == 2:
        if type(k) == int and k > 0 and k < X.shape[0]:
            d = X.shape[1]
            centroids = np.random.uniform(
                np.amin(X, axis=0),
                np.amax(X, axis=0),
                (k, d)
            )
            return centroids
        else:
            return None
    else:
        return None
