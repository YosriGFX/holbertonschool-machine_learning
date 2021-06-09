#!/usr/bin/env python3
'''0. Likelihood'''
import numpy as np


def intersection(x, n, P, Pr):
    '''Calculates the intersection of obtaining this
    data with the various hypothetical probabilities'''
    if isinstance(n, (int, float)) and n > 0:
        if isinstance(x, (int, float)) and x >= 0:
            if x <= n:
                if isinstance(P, np.ndarray)\
                  and len(P.shape) == 1 and P.shape[0] >= 1:
                    if np.any(P > 1) or np.any(P < 0):
                        raise ValueError(
                            'All values in P must be in the range [0, 1]'
                        )
                    else:
                        if np.any(Pr > 1) or np.any(Pr < 0):
                            raise ValueError(
                                'All values in Pr must be in the range [0, 1]'
                            )
                        else:
                            if np.isclose(np.sum(Pr), 1):
                                com = np.math.factorial(
                                    n
                                ) / (
                                    np.math.factorial(
                                        x
                                    ) * np.math.factorial(n-x)
                                )
                                return (
                                    com * pow(
                                        P, x
                                    ) * pow(
                                        1 - P,
                                        n - x
                                    )
                                ) * Pr
                            else:
                                raise ValueError('Pr must sum to 1')
                else:
                    raise TypeError('P must be a 1D numpy.ndarray')
            else:
                raise ValueError('x cannot be greater than n')
        else:
            raise ValueError(
                'x must be an integer that is greater than or equal to 0'
            )
    else:
        raise ValueError("n must be a positive integer")
