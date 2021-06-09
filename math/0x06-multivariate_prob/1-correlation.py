#!/usr/bin/env python3
'''Mean and Covariance'''
import numpy as np


def correlation(C):
    '''Calculates a correlation matrix'''
    if type(C) is np.ndarray:
        if len(C.shape) == 2 and C.shape[0] == C.shape[1]:
            d = np.sqrt(
                np.diag(C)
            )
            Outer = np.outer(
                d, d
            )
            Correlation = C / Outer
            Correlation[C == 0] = 0
            return Correlation
        else:
            raise ValueError("C must be a 2D square matrix")
    else:
        raise TypeError("C must be a numpy.ndarray")
