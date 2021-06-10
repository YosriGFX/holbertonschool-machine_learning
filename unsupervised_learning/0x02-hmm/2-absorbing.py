#!/usr/bin/env python3
'''2. Absorbing Chains'''
import numpy as np


def validator(P):
    '''validator'''
    if not isinstance(P, np.ndarray) or len(P.shape) != 2:
        return False
    if P.shape[0] != P.shape[1] or np.sum(P, axis=1).all() != 1:
        return False
    if all(np.diag(P) == 1):
        return True
    if not any(np.diag(P) == 1):
        return False
    return 'continue'


def absorbing(P):
    '''determines if a markov chain is absorbing'''
    checker = validator(P)
    if type(checker) == bool:
        return checker
    else:
        a = np.where(np.diag(P) == 1)
        b = np.sum(P[a[0]], axis=0)
        for i in range(P.shape[0]):
            inter = b * (P[i] != 0)
            if (inter == 1).any():
                b[i] = 1
        return b.all()
