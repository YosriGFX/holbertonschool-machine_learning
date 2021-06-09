#!/usr/bin/env python3
'''1. K-means'''
import numpy as np


def kmeans(X, k, iterations=1000):
    '''performs K-means on a dataset'''
    if isinstance(X, np.ndarray) and len(X.shape) == 2:
        if type(k) == int and k > 0 and X.shape[0] >= k:
            if type(iterations) == int and iterations > 0:
                low = np.amin(X, axis=0)
                high = np.amax(X, axis=0)
                n, d = X.shape
                C = np.random.uniform(low, high, (k, d))
                old_C = np.copy(C)
                a = np.tile(X, k).reshape(n, k, d)
                tmp = C.reshape(-1)
                b = np.tile(tmp, (n, 1)).reshape(n, k, d)
                e = a-b
                dest = np.linalg.norm(e, axis=2)
                clss = np.argmin(dest, axis=1)
                for i in range(iterations):
                    for j in range(k):
                        data_indx = np.where(clss == j)
                        if len(data_indx[0]) == 0:
                            C[j] = np.random.uniform(low, high, (1, d))
                        else:
                            C[j] = np.mean(X[data_indx], axis=0)
                    a = np.tile(X, k).reshape(n, k, d)
                    tmp = C.reshape(-1)
                    b = np.tile(tmp, (n, 1)).reshape(n, k, d)
                    e = a-b
                    dest = np.linalg.norm(e, axis=2)
                    clss = np.argmin(dest, axis=1)
                    if (C == old_C).all():
                        return C, clss
                    old_C = np.copy(C)
                return C, clss
            else:
                return None, None
        else:
            return None, None
    else:
        return None, None
