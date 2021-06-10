#!/usr/bin/env python3
'''4. The Viretbi Algorithm'''
import numpy as np


def validator(Observation, Emission, Transition, Initial):
    '''validator'''
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


def viterbi(Observation, Emission, Transition, Initial):
    '''calculates the most likely sequence
    of hidden states for a hidden markov model'''
    if validator(Observation, Emission, Transition, Initial):
        N = Emission.shape[0]
        M = Observation.shape[0]
        Viterbi = np.zeros((N, M))
        a = np.zeros((N, M))
        a[:, 0] = 0
        Viterbi[:, 0] = np.multiply(
            Initial[:, 0], Emission[:, Observation[0]]
        )
        for M in range(1, M):
            b = Viterbi[:, M - 1] * Transition.T
            c = np.amax(b, axis=1)
            Viterbi[:, M] = c * Emission[:, Observation[M]]
            a[:, M - 1] = np.argmax(b, axis=1)
        path = [np.argmax(Viterbi[:, M - 1])] + []
        tmp = np.argmax(Viterbi[:, M - 1])
        for M in range(M - 2, -1, -1):
            tmp = int(a[tmp, M])
            path = [tmp] + path
        P = np.amax(Viterbi[:, M - 1], axis=0)
        return path, P
    else:
        return None, None
