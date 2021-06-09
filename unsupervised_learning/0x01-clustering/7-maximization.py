#!/usr/bin/env python3
'''7. Maximization'''
import numpy as np


def maximization(X, g):
    '''calculates the maximization step
    in the EM algorithm for a GMM'''
    try:
        if isinstance(X, np.ndarray) and len(X.shape) == 2:
            if isinstance(g, np.ndarray) and len(X.shape) == 2:
                n, d = X.shape
                if n != g.shape[1]:
                    return None, None, None
                tester = np.ones((n, ))
                if not np.isclose(
                    np.sum(g, axis=0), tester
                ).all():
                    return None, None, None
                pi = np.zeros((g.shape[0],))
                m = np.zeros((g.shape[0], d))
                S = np.zeros((g.shape[0], d, d))
                for a in range(g.shape[0]):
                    b = np.sum(g[a])
                    pi[a] = b / n
                    m[a] = np.sum(
                        np.matmul(
                            g[a].reshape(1, n),
                            X
                        ), axis=0
                    ) / b
                    c = (X - m[a])
                    S[a] = np.dot(
                        g[a].reshape(1, n) * c.T, c
                    ) / b
                return pi, m, S
            else:
                return None, None, None
        else:
            return None, None, None
    except Exception:
        return None, None, None
