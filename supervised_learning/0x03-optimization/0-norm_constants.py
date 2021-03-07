#!/usr/bin/env python3
'''Normalization Constants'''
import numpy as np


def normalization_constants(X):
    '''Function that calculates the normalization
    (standardization) constants of a matrix'''
    return X.mean(
        axis=0
    ), X.std(
        axis=0
    )
