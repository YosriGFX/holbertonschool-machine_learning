#!/usr/bin/env python3
'''Matisse'''


def poly_derivative(poly):
    '''poly_derivative function'''
    if type(poly) is list and len(poly) > 0:
        derivative = []
        for i in range(0, len(poly)):
            if not isinstance(poly[i], (int, float)):
                return None
            derivative.append(poly[i] * i)
        if len(poly) == 0 or sum(derivative) == 0:
            return [0]
        del derivative[0]
        return derivative
    else:
        return None
