#!/usr/bin/env python3
'''0. From Numpy'''
import pandas as pd
import string


def from_numpy(array):
    '''A function that creates a
    pd.DataFrame from a np.ndarray'''
    return pd.DataFrame(
        array,
        columns=[
            string.ascii_uppercase[i] for i in range(array.shape[1])
        ]
    )
