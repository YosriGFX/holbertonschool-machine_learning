#!/usr/bin/env python3
'''Ridin Bareback'''


def mat_mul(mat1, mat2):
    '''mat mul'''
    if len(mat1[0]) != len(mat2):
        return None
    else:
        return [
            [
                mat1[a][c] * mat2[c][b]
                + mat1[a][c + 1] * mat2[c + 1][b]
                for b in range(len(mat2[0]))
                for c in range(len(mat1[0]) - 1)
            ]
            for a in range(len(mat1))
        ]
