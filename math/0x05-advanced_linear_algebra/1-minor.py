#!/usr/bin/env python3
'''Determinant'''


def matrix_validator(matrix):
    '''matrix Validator'''
    if len(matrix) > 0:
        if type(matrix) is list\
          and all([type(mat) is list for mat in matrix]):
            if all([len(mat) == len(matrix) for mat in matrix]):
                return True
            else:
                raise ValueError('matrix must be a non-empty square matrix')
        else:
            raise TypeError('matrix must be a list of lists')
    else:
        raise TypeError('matrix must be a list of lists')


def new_m(matrix, a, b):
    '''new_m'''
    return [
        [
            matrix[i][j] for j
            in range(len(matrix[i])) if j != a
        ]
        for i in range(len(matrix)) if i != b
    ]


def determinant(matrix):
    '''Calculates the determinant of a matrix'''
    if len(matrix[0]) == 0:
        return 1
    elif len(matrix) == 1:
        return matrix[0][0]
    elif len(matrix) == 2:
        return matrix[0][0] * matrix[1][1]\
          - matrix[0][1] * matrix[1][0]
    else:
        Determinant = 0
        for a in range(len(matrix[0])):
            Determinant += matrix[0][a] * determinant(
                new_m(matrix, a, 0)
            ) * ((-1) ** a)
        return Determinant


def minor(matrix):
    '''Calculates the minor matrix of a matrix'''
    if matrix_validator(matrix):
        if len(matrix) == 1 and len(matrix[0]) == 1:
            return [[1]]
        return [
            [
                determinant(
                    new_m(matrix, j, i)
                ) for j in range(len(matrix[i]))
            ] for i in range(len(matrix))
        ]
