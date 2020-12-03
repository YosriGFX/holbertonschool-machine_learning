#!/usr/bin/env python3
'''Cat Got your tongue'''
import numpy as np


def np_cat(mat1, mat2, axis=0):
    '''np_cat'''
    return np.append(mat1, mat2, axis)
