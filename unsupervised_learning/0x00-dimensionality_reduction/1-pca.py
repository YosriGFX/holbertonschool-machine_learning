#!/usr/bin/env python3
'''1. PCA'''
import numpy as np


def pca(X, ndim):
    '''performs PCA on a dataset'''
    Mean = X - np.mean(X, axis=0)
    return np.dot(
        Mean,
        np.linalg.svd(Mean)[2].T[:, :ndim]
    )
