#!/usr/bin/env python3
'''0. Markov Chain'''
import numpy as np


def validator(P, s, t=1):
    '''validator'''
    if not isinstance(P, np.ndarray) or len(P.shape) != 2:
        return None
    if P.shape[0] != P.shape[1] or np.sum(P, axis=1).all() != 1:
        return None
    if not isinstance(s, np.ndarray) or len(s.shape) != 2:
        return None
    if s.shape[0] != 1 or s.shape[1] != P.shape[0]:
        return None
    if not isinstance(t, int) or t < 0:
        return None
    return True


def markov_chain(P, s, t=1):
    '''determines the probability of a markov
    chain being in a particular state after
    a specified number of iterations'''
    if validator(P, s, t):
        if t != 0:
            T = np.matmul(s, P)
            for a in range(1, t):
                T = np.matmul(T, P)
            return T
        else:
            return s
