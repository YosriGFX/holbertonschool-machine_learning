#!/usr/bin/env python3
'''Integrate'''


def poly_integral(poly, C=0):
    '''Calculates the integral of a polynomia'''
    if (
        type(poly) != list
    ) or (
        len(poly) == 0
    ) or (
        type(C) != int
    ):
        return None
    else:
        integral = [C]
        if (len(poly) == 1) and poly[0] == 0:
            return integral
        for i in range(0, len(poly)):
            if isinstance(poly[i], (int, float)):
                integ = poly[i] / (i + 1)
                integ = integ.__trunc__() if not integ % 1 else float(integ)
                integral.append(integ)
            else:
                return None

        return integral
