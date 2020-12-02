#!/usr/bin/env python3
'''Matrix Transpose'''


def matrix_transpose(matrix):
    '''matrix transpose'''
    new_matrix = []
    for j in range(len(matrix[0])):
        new_matrix.append(
            [matrix[i][j] for i in range(len(matrix))]
        )
    return new_matrix
