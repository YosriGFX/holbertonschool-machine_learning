#!/usr/bin/env python3
'''Gettin Cozy'''


def cat_matrices2D(mat1, mat2, axis=0):
    '''cat matrice2D'''
    if axis == 0:
        return [
            mat1[i] for i in range(len(mat1))
        ] + [
            mat2[i] for i in range(len(mat2))
        ]
    elif axis == 1:
        return [
            mat1[i] + mat2[i]
            for i in range(len(mat1))
        ]
    else:
        return None
