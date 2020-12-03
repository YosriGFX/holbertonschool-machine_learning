#!/usr/bin/env python3
'''Gettin Cozy'''


def cat_matrices2D(mat1, mat2, axis=0):
    '''cat matrice2D'''
    if not axis:
        if len(mat1[0]) != len(mat2[0]):
            return None
        return [i.copy() for i in mat1] + [i.copy() for i in mat2]
    else:
        if len(mat1) != len(mat2):
            return None
        return [mat1[i].copy() + mat2[i].copy() for i in range(len(mat1))]
