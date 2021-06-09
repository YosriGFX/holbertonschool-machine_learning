#!/usr/bin/env python3
'''4. Initialize GMM'''
import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    '''initializes variables for a Gaussian Mixture Model'''
    if isinstance(X, np.ndarray) and len(X.shape) == 2:
        if type(k) == int and k > 0 and X.shape[0] >= k:
            pi = np.full(
                (k,),
                1 / k
            )
            m = kmeans(X, k)[0]
            S = np.tile(
                np.identity(
                    X.shape[1]
                ), (
                    k, 1
                )
            ).reshape(
                k,
                X.shape[1],
                X.shape[1]
            )
            return pi, m, S
        else:
            return None, None, None
    else:
        return None, None, None
