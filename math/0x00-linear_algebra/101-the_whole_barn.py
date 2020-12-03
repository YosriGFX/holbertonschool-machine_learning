#!/usr/bin/env python3
'''The Whole Barn'''


def add_matrices(mat1, mat2):
    '''add matrices'''
    try:
        if len(mat1) != len(mat2):
            raise ValueError
        if isinstance(mat1[0], list) and isinstance(mat2[0], list):
            if list(map(add_matrices, mat1, mat2))[0]:
                return list(map(add_matrices, mat1, mat2))
            else:
                return None
        else:
            return [a + b for a, b in zip(mat1, mat2)]
    except ValueError:
        return None
