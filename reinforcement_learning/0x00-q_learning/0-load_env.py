#!/usr/bin/env python3
'''0. Load the Environment'''
import numpy as np
import gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    '''that loads the pre-made FrozenLakeEnv evnironment from OpenAI s gym'''
    return gym.make(
        'FrozenLake-v0',
        map_name=map_name,
        desc=desc,
        is_slippery=is_slippery
    )
