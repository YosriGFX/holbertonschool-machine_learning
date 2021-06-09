#!/usr/bin/env python3
'''4. Continuous Posterior'''
import numpy as np
from scipy import special


def posterior(x, n, p1, p2):
    '''calculates the posterior probability that
    the probability of developing severe side effects
    falls within a specific range given the data'''
    if isinstance(n, int) and n >= 1:
        if isinstance(x, int) or x >= 0:
            if x <= n:
                if isinstance(p1, float) and p1 >= 0 and p1 <= 1:
                    if isinstance(p2, float) and p2 >= 0 and p2 <= 1:
                        if p2 > p1:
                            a = special.betainc(
                                x + 1,
                                n - x + 1,
                                p1
                            )
                            b = special.betainc(
                                x + 1,
                                n - x + 1,
                                p2
                            )
                            return b - a
                        else:
                            raise ValueError('p2 must be greater than p1')
                    else:
                        raise ValueError(
                            'p2 must be a float in the range [0, 1]'
                        )
                else:
                    raise ValueError('p1 must be a float in the range [0, 1]')
            else:
                raise ValueError('x cannot be greater than n')
        else:
            raise ValueError(
                'x must be an integer that is greater than or equal to 0'
            )
    else:
        raise ValueError('n must be a positive integer')
