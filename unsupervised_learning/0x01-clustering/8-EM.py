#!/usr/bin/env python3
'''8. EM'''
import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    '''performs the expectation maximization for a GMM:'''
    if isinstance(X, np.ndarray) and len(X.shape) == 2:
        if type(k) == int and k > 0 and X.shape[0] >= k:
            if type(iterations) == int and iterations > 0:
                if type(tol) == float and tol >= 0:
                    if type(verbose) == bool:
                        for a in range(iterations):
                            if a == 0:
                                tmp = 0
                                pi, m, S = initialize(X, k)
                            else:
                                pi, m, S = maximization(X, g)
                            g, L = expectation(X, pi, m, S)
                            if verbose:
                                if a % 10 == 0\
                                    or a == iterations-1\
                                        or abs(L-tmp) <= tol:
                                    print(
                                        'Log Likelihood after \
{} iterations: {}'.format(
                                            a, L
                                        )
                                    )
                            if abs(L-tmp) <= tol:
                                break
                            tmp = L

                        return pi, m, S, g, L
                    else:
                        return None, None, None, None, None
                else:
                    return None, None, None, None, None
            else:
                return None, None
        else:
            return None, None
    else:
        return None, None
