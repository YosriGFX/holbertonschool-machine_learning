#!/usr/bin/env python3
'''2. Epsilon Greedy'''
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    '''A function that uses epsilon-greedy
    to determine the next action'''
    if np.random.uniform(0, 1) > epsilon:
        action = np.argmax(Q[state, :])
    else:
        action = np.random.randint(0, Q.shape[1])
    return action
