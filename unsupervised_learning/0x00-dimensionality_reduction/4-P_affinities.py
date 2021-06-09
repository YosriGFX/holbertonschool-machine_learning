#!/usr/bin/env python3
'''4. P affinities'''
import numpy as np
P_init = __import__('2-P_init').P_init
HP = __import__('3-entropy').HP


def P_affinities(X, tol=1e-5, perplexity=30.0):
    '''calculates the symmetric P affinities of a data set'''
    a = X.shape[0]
    D, P, betas, H = P_init(X, perplexity)
    if a != 0:
        for i in range(a):
            row = D[i].copy()
            row = np.delete(row, i, axis=0)
            Hi, Pi = HP(row, betas[i])
            Hd = Hi - H
            Max = None
            Min = None
            while np.abs(Hd) > tol:
                if Hd < 0:
                    Max = betas[i, 0]
                    if Min is None:
                        betas[i, 0] = betas[
                            i, 0
                        ] / 2.
                    else:
                        betas[i, 0] = (
                            betas[i, 0] + Min
                        ) / 2.
                else:
                    Min = betas[i, 0]
                    if Max is None:
                        betas[i, 0] = betas[
                            i, 0
                        ] * 2.
                    else:
                        betas[i, 0] = (
                            betas[i, 0] + Max
                        ) / 2.
                Hi, Pi = HP(row, betas[i])
                Hd = Hi - H
            Pi = np.insert(Pi, i, 0)
            P[i] = Pi
        P = (
            P.T + P
        ) / (
            2 * a
        )
        return P
    else:
        return P
