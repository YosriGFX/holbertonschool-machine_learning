#!/usr/bin/env python3
"""HMM module"""
import numpy as np


def validator(Observations, Transition, Emission, Initial, iterations=1000):
    '''validator'''
    if not isinstance(Observations, np.ndarray) or\
            len(Observations.shape) != 1:
        return False
    if not isinstance(Emission, np.ndarray) or len(Emission.shape) != 2:
        return False

    T, = Observations.shape
    N, M = Emission.shape

    if not isinstance(Transition, np.ndarray) or Transition.shape != (N, N):
        return False
    if np.any(np.sum(Transition, axis=1) != 1):
        return False
    if not isinstance(Initial, np.ndarray) or Initial.shape != (N, 1):
        return False
    if np.sum(Initial) != 1:
        return False
    if not isinstance(iterations, int) or iterations < 1:
        return False
    return True


def forward(Observation, Emission, Transition, Initial):
    '''performs the forward algorithm for a hidden markov model'''
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


def backward(Observation, Emission, Transition, Initial):
    '''performs the backward algorithm for a hidden markov model'''
    N = Emission.shape[0]
    T = Observation.shape[0]
    B = np.zeros((N, T))
    B[:, T - 1] = np.ones(N)
    for t in range(T - 2, -1, -1):
        B[:, t] = np.sum(
            Transition * Emission[:, Observation[
                t + 1
            ]] * B[:, t + 1], axis=1
        )
    P = np.sum(Initial[:, 0] * Emission[:, Observation[
        0
    ]] * B[:, 0])
    return P, B


def expectation(Observations, Transition, Emission, forward, backward):
    '''performs the Expectation step'''
    T, = Observations.shape
    N, M = Emission.shape
    tmp, F = forward
    B = backward[1]
    xi = np.zeros((N, N, T - 1))
    for t in range(T - 1):
        obs = Observations[t + 1]
        for i in range(N):
            for j in range(N):
                xi[i, j, t] = (
                    F[i, t] *
                    Transition[i, j] *
                    Emission[j, obs] *
                    B[j, t + 1]
                ) / tmp
    gamma = (F * B) / tmp
    return gamma, xi


def maximization(Observations, gamma, xi, dimension):
    '''performs the Maximization step'''
    N, M = dimension
    T = Observations.shape[0]
    Transition = np.sum(xi, axis=2) / np.sum(
        gamma[:, :T-1], axis=1
        )[..., np.newaxis]
    numerator = np.zeros((N, M))
    for k in range(M):
        numerator[:, k] = np.sum(gamma[:, Observations == k], axis=1)
    Emission = numerator / np.sum(gamma, axis=1)[..., np.newaxis]
    return Transition, Emission


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    '''performs the Baum-Welch algorithm for a hidden markov model'''
    if validator(Observations, Transition, Emission, Initial, iterations):
        T, = Observations.shape
        N, M = Emission.shape
        for i in range(iterations):
            gamma, xi = expectation(
                Observations,
                Transition,
                Emission,
                forward(Observations, Emission, Transition, Initial),
                backward(Observations, Emission, Transition, Initial)
            )
            Transition, Emission = maximization(
                Observations, gamma, xi, (N, M)
            )
        return Transition, Emission
    else:
        return None, None
