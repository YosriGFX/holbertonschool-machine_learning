#!/usr/bin/env python3
'''Squashed Like Sardines'''
import numpy as np


def cat_matrices(mat1, mat2, axis=0):
    '''cat matrices'''
    try:
        return np.concatenate((mat1, mat2), axis).tolist()
    except ValueError:
        return None
