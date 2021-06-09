#!/usr/bin/env python3
'''6. Expectation'''
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    '''calculates the expectation step
    in the EM algorithm for a GMM'''
    if isinstance(X, np.ndarray) and len(X.shape) == 2:
        if isinstance(pi, np.ndarray) and len(pi.shape) == 1:
            if isinstance(m, np.ndarray) and len(m.shape) == 2:
                if isinstance(S, np.ndarray) and len(S.shape) == 3:
                    a, d = X.shape
                    k = pi.shape[0]
                    if k > a:
                        return None, None
                    if d != m.shape[1] or d != S.shape[1] or d != S.shape[2]:
                        return None, None
                    if k != m.shape[0] or k != S.shape[0]:
                        return None, None
                    if not np.isclose([np.sum(pi)], [1])[0]:
                        return None, None
                    Sum = 0
                    g = np.zeros((k, a))
                    for b in range(k):
                        tmp = pi[b]*pdf(X, m[b], S[b])
                        g[b] = tmp
                        Sum += tmp
                    g /= Sum
                    L = np.sum(np.log(Sum))
                    return g, L
                else:
                    return None, None
            else:
                return None, None
        else:
            return None, None
    else:
        return None, None
