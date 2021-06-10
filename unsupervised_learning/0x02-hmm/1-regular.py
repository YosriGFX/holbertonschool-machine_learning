#!/usr/bin/env python3
'''1. Regular Chains'''
import numpy as np


def validator(P):
    '''Validator'''
    if not isinstance(P, np.ndarray) or len(P.shape) != 2:
        return None
    if P.shape[0] != P.shape[1] or np.sum(P, axis=1).all() != 1:
        return None
    return True


def regular(P):
    '''determines the steady state
    probabilities of a regular markov chain'''
    if validator(P):
        a, b = np.linalg.eig(P.T)
        temp = np.where(np.isclose(a, 1))
        temp = temp[0][0] if len(temp[0]) else None
        if temp is None:
            return None
        else:
            steady = b[:, temp]
            if any(np.isclose(steady, 0)):
                return None
            else:
                steady = steady / np.sum(steady)
                return steady[np.newaxis, :]
