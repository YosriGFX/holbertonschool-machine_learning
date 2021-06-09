#!/usr/bin/env python3
'''5. PDF'''
import numpy as np


def pdf(X, m, S):
    '''calculates the probability density
    function of a Gaussian distribution'''
    if isinstance(X, np.ndarray) and len(X.shape) == 2:
        if isinstance(m, np.ndarray) and len(m.shape) == 1:
            if isinstance(S, np.ndarray) and len(S.shape) == 2:
                if X.shape[1] == m.shape[0] and X.shape[1] == S.shape[0]:
                    if S.shape[0] == S.shape[1]:
                        Det = np.linalg.det(S)
                        Inv = np.linalg.inv(S)
                        P = np.exp(
                            -np.einsum(
                                '...k,kl,...l->...',
                                X - m,
                                Inv,
                                X - m
                            ) / 2
                        ) / np.sqrt(
                            (
                                2 * np.pi
                            ) ** X.shape[1] * Det
                        )
                        P = np.where(P < 1e-300, 1e-300, P)
                        return P
                    else:
                        return None
                else:
                    return None
            else:
                return None
        else:
            return None
    else:
        return None
