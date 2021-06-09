#!/usr/bin/env python3
'''3. Optimize k'''
import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    '''tests for the optimum number of clusters by variance'''
    try:
        if isinstance(X, np.ndarray) and len(X.shape) == 2:
            if kmax is None:
                kmax = X.shape[0]
            if type(kmin) == int and kmin > 0 and X.shape[0] > kmin:
                if type(kmax) == int and kmax > 0 and X.shape[0] >= kmax:
                    if kmax > kmin:
                        if type(iterations) == int and iterations > 0:
                            results = []
                            Variance_K = []
                            for k in range(kmin, kmax+1):
                                C, clss = kmeans(X, k, iterations)
                                results.append((C, clss))
                                Variance = variance(X, C)
                                Variance_K.append(Variance)
                            d_vars = [
                                Variance_K[
                                    0
                                ] - Variance for Variance in Variance_K
                            ]
                            return results, d_vars
                        else:
                            return None, None
                    else:
                        return None, None
                else:
                    return None, None
            else:
                return None, None
        else:
            return None, None
    except Exception:
        return None, None
