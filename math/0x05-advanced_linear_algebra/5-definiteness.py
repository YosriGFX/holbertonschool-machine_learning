#!/usr/bin/env python3
'''Definiteness'''
import numpy as np


def definiteness(matrix):
    '''Calculates the definiteness of a matrix'''
    if type(matrix) is np.ndarray:
        if len(matrix.shape) == 2:
            if np.all(matrix.T == matrix):
                if len(matrix.shape) != 2\
                  or matrix.shape[0] == matrix.shape[1]:
                    if np.all(np.linalg.eigvals(matrix) > 0):
                        return "Positive definite"
                    elif np.all(np.linalg.eigvals(matrix) >= 0):
                        return "Positive semi-definite"
                    elif np.all(np.linalg.eigvals(matrix) < 0):
                        return "Negative definite"
                    elif np.all(np.linalg.eigvals(matrix) <= 0):
                        return "Negative semi-definite"
                    else:
                        return "Indefinite"
                else:
                    return None
            else:
                return None
        else:
            return None
    else:
        raise TypeError("matrix must be a numpy.ndarray")
