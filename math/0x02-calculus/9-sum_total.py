#!/usr/bin/env python3
'''Sum'''


def summation_i_squared(n):
    '''summation_i_squared'''
    if (type(n) != int) or (n < 1):
        return None
    return sum(map(lambda a: a ** 2, range(n + 1)))
