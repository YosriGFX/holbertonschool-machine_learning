#!/usr/bin/env python3
'''1. Initialize Q-table'''


def q_init(env):
    '''initializes the Q-table'''
    return np.zeros(
        (
            env.observation_space.n,
            env.action_space.n
        )
    )
