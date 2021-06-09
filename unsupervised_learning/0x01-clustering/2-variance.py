#!/usr/bin/env python3
'''2. Variance'''
import numpy as np


def variance(X, C):
    '''calculates the total intra-cluster variance for a data set'''
    try:
        if isinstance(X, np.ndarray) and len(X.shape) == 2:
            if isinstance(C, np.ndarray) and len(X.shape) == 2:
                n, d = X.shape
                k = C.shape[0]
                if d == C.shape[1]:
                    if k <= X.shape[0]:
                        var = np.sum(
                            np.sum(
                                (
                                    np.min(
                                        np.linalg.norm(
                                            np.tile(
                                                X, k
                                            ).reshape(
                                                n, k, d
                                            ) - np.tile(
                                                C.reshape(-1),
                                                (n, 1)
                                            ).reshape(
                                                n, k, d
                                            ), axis=2
                                        ), axis=1
                                    )
                                )**2
                            )
                        )
                        return var
                    else:
                        return None
                else:
                    return None
            else:
                return None
        else:
            return None
    except Exception:
        return None
