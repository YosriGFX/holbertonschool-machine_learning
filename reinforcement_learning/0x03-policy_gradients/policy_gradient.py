#!/usr/bin/env python3
'''1. Compute the Monte-Carlo policy gradient'''
import numpy as np


def soft_max(value):
    '''A function that Computes the softmax'''
    exp = np.exp(
        value - np.max(value)
    )
    return exp / np.sum(exp)


def policy(matrix, weight):
    '''A function that computes to policy
    with a weight of a matrix'''
    return soft_max(matrix @ weight)


def softmax_grad(softmax):
    '''A function that computes the gradient
    of a given softmax'''
    softmax = softmax.reshape(-1, 1)
    return np.diagflat(softmax) - softmax @ softmax.T


def policy_gradient(state, weight):
    '''A function that computes the Monte-Carlo policy
    gradient based on a state and a weight matrix.'''
    Pi = policy(state, weight)
    action = np.random.choice(len(Pi[0]), p=Pi[0])
    log = softmax_grad(Pi)[action, :] / Pi[0, action]
    gradient = state.T @ log[None, :]
    return action, gradient
