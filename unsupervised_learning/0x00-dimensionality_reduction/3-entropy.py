#!/usr/bin/env python3
'''3. Entropy'''
import numpy as np


def HP(Di, beta):
    '''calculates the Shannon entropy and
    P affinities relative to a data point'''
    P = np.exp(
        -Di * beta
    )
    Sum = np.sum(P)
    Pi = P / Sum
    Hi = -np.sum(
        Pi * np.log2(Pi)
    )
    return (Hi, Pi)
