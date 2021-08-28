#!/usr/bin/env python3
'''4. Play'''
import numpy as np


def play(env, Q, max_steps=100):
    '''A function that has the trained agent play an episode'''
    rewards = []
    state = env.reset()
    done = False
    total_rewards = 0
    for step in range(max_steps):
        env.render()
        action = np.argmax(Q[state])
        new_state, reward, done, _ = env.step(action)
        total_rewards += reward
        if done:
            env.render()
            rewards.append(total_rewards)
            break
        state = new_state
    return total_rewards
