#!/usr/bin/env python3
'''5. Q affinities'''
import numpy as np


def Q_affinities(Y):
    '''calculates the Q affinities'''
    num = -2. * np.dot(Y, Y.T)
    num = 1. / (
        1. + np.add(
            np.add(
                (
                    -2. * np.dot(
                        Y, Y.T
                    )
                ),
                np.sum(
                    np.square(Y), 1
                )
            ).T,
            np.sum(
                np.square(Y), 1
            )
        )
    )
    np.fill_diagonal(num, 0)
    Q = num / np.sum(num)
    return Q, num
