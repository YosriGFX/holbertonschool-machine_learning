#!/usr/bin/env python3
'''One hot encoder'''
import numpy as np


def one_hot_decode(one_hot):
    '''converts a one-hot matrix
    into a vector of labels'''
    if (
        type(one_hot) is np.ndarray
    ) and (
        len(one_hot.shape) is 2
    ):
        try:
            return np.argmax(one_hot, axis=0)
        except Exception:
            return None
    else:
        return None
