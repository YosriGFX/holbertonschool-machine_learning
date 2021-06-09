#!/usr/bin/env python3
'''0. Likelihood'''
import numpy as np


def validator(x, n, P, Pr):
    '''Calculates the marginal probability of obtaining the data'''
    if isinstance(n, (int, float)) and n > 0:
        if isinstance(x, (int, float)) and x >= 0:
            if x <= n:
                if isinstance(P, np.ndarray)\
                  and len(P.shape) == 1 and P.shape[0] >= 1:
                    if isinstance(Pr, np.ndarray) and Pr.shape == P.shape:
                        if np.any(P > 1) or np.any(P < 0):
                            raise ValueError('All values \
in P must be in the range [0, 1]')
                        else:
                            if np.any(Pr > 1) or np.any(Pr < 0):
                                raise ValueError(
                                    'All values in Pr must \
be in the range [0, 1]'
                                )
                            else:
                                if np.isclose(np.sum(Pr), 1):
                                    return True
                                else:
                                    raise ValueError('Pr must sum to 1')
                    else:
                        raise TypeError('Pr must be a numpy.ndarray \
with the same shape as P')
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


def intersection(x, n, P, Pr):
    '''Calculates the intersection of obtaining this
    data with the various hypothetical probabilities'''
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


def marginal(x, n, P, Pr):
    '''Calculates the marginal probability of obtaining the data'''
    return np.sum(
        intersection(x, n, P, Pr)
    )


def posterior(x, n, P, Pr):
    '''Calculates the posterior probability for the various
    hypothetical probabilities of developing severe side
    effects given the data'''
    if validator(x, n, P, Pr):
        return intersection(
            x, n, P, Pr
        ) / marginal(
            x, n, P, Pr
        )
