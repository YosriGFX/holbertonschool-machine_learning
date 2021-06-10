#!/usr/bin/env python3
'''9. BIC'''
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    '''finds the best number of clusters for a GMM
    using the Bayesian Information Criterion'''
    try:
        if isinstance(X, np.ndarray) and len(X.shape) == 2:
            if type(kmin) == int and kmin > 0 and X.shape[0] > kmin:
                if type(kmax) == int and kmax > 0 and X.shape[0] > kmax:
                    if type(iterations) == int and iterations > 0:
                        if type(tol) == float and tol >= 0:
                            if type(verbose) == bool:
                                n, d = X.shape
                                L = np.empty((kmax - kmin + 1, ))
                                b = np.empty((kmax - kmin + 1, ))
                                a = []
                                c = []
                                e = []
                                for i in range(kmin, kmax + 1):
                                    pi, m, S, g, tmp = \
                                        expectation_maximization(
                                            X,
                                            i,
                                            iterations,
                                            tol,
                                            verbose
                                        )
                                    a.append(pi)
                                    c.append(m)
                                    e.append(S)
                                    L[i - 1] = tmp
                                    p = (i * d * (d + 1) / 2) + (d * i) + i - 1
                                    b[i - 1] = p * np.log(n) - 2 * tmp
                                best_k = np.argmin(b)
                                best_result = (
                                    a[best_k],
                                    c[best_k],
                                    e[best_k]
                                )
                                best_k = best_k + 1
                                return best_k, best_result, L, b
                            else:
                                return None, None, None, None
                        else:
                            return None, None, None, None
                    else:
                        return None, None, None, None
                else:
                    return None, None, None, None
            else:
                return None, None, None, None
        else:
            return None, None, None, None
    except Exception:
        return None, None, None, None
