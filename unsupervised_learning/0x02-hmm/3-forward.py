#!/usr/bin/env python3
'''3. The Forward Algorithm'''
import numpy as np


def validator(Observation, Emission, Transition, Initial):
    '''Validator'''
    if not isinstance(Observation, np.ndarray) or len(Observation.shape) != 1:
        return False
    if not isinstance(Emission, np.ndarray) or len(Emission.shape) != 2:
        return False
    if not isinstance(Transition, np.ndarray) or len(Transition.shape) != 2:
        return False
    if not isinstance(Initial, np.ndarray) or len(Initial.shape) != 2:
        return False
    if Transition.shape[0] != Emission.shape[
        0
    ] or Transition.shape[1] != Emission.shape[0]:
        return False
    if Initial.shape[0] != Emission.shape[0] or Initial.shape[1] != 1:
        return False
    if not np.sum(Emission, axis=1).all():
        return False
    if not np.sum(Transition, axis=1).all() or not np.sum(Initial) == 1:
        return False
    return True


def forward(Observation, Emission, Transition, Initial):
    '''performs the forward algorithm for a hidden markov model'''
    if validator(Observation, Emission, Transition, Initial):
        N = Emission.shape[0]
        M = Observation.shape[0]
        F = np.zeros((N, M))
        F[:, 0] = np.multiply(Initial[:, 0], Emission[:, Observation[0]])
        for i in range(1, M):
            prob = np.multiply(
                np.matmul(
                    F[:, i - 1], Transition
                ), Emission[:, Observation[i]]
            )
            F[:, i] = prob
        P = np.sum(F[:, M - 1])
        return P, F
    else:
        return None, None
