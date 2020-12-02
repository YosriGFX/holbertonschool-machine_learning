#!/usr/bin/env python3
'''Across the Planes'''


def add_matrices2D(mat1, mat2):
    '''add matrice3D'''
    if len(mat1[0]) != len(mat2[0]):
        return None
    else:
        return [
            [
                mat1[i][j] + mat2[i][j]
                for j in range(len(mat1[0]))
            ]
            for i in range(len(mat1))
        ]
