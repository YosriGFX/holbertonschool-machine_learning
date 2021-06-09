#!/usr/bin/env python3
'''0. PCA'''
import numpy as np


def pca(X, var=0.95):
    '''performs PCA on a dataset'''
    Sum = np.cumsum(
        np.linalg.svd(X)[1]
    )
    Sum = Sum / Sum[-1]
    result = np.min(
        np.where(Sum >= var)
    )
    variance = np.linalg.svd(X)[2].T
    variance_res = variance[..., :result + 1]
    return variance_res
