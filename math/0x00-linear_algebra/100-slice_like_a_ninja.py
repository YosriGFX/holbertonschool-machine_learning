#!/usr/bin/env python3
'''Slice like a ninja'''


def np_slice(matrix, axes={}):
    '''np slice'''
    return matrix[
        tuple(
            (
                slice(*axes.get(a, (None, None)))
                for a in range(len(matrix.shape))
            )
        )
    ]
