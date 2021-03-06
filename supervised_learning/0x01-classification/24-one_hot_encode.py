#!/usr/bin/env python3
'''One hot encoder'''
import numpy as np


def one_hot_encode(Y, classes):
    '''converts a numeric label vector into a one-hot matrix'''
    if (
        Y is not None
    ) and (
        type(Y) is np.ndarray
    ) and (
        type(classes) is int
    ):
        try:
            oneMatrix = np.zeros((classes, Y.shape[0]))
            oneMatrix[Y, np.arange(Y.shape[0])] = 1
            return oneMatrix
        except Exception:
            return None
    else:
        return None
