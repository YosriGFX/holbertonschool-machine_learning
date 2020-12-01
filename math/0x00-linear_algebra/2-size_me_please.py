#!/usr/bin/env python3
'''Matrix Shape'''


def matrix_shape(matrix):
    '''Matrix Shape'''
    if type(matrix[0][0]) is list:
        return [len(matrix)] + matrix_shape(matrix[0])
    else:
        return [len(matrix), len(matrix[0])]
