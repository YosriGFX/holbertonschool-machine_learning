#!/usr/bin/env python3
'''1. TD(λ)'''
import numpy as np


def td_lambtha(
    env,
    V,
    policy,
    lambtha,
    episodes=5000,
    max_steps=100,
    alpha=0.1,
    gamma=0.99
):
    '''A function that performs the TD(λ) algorithm'''
    for _ in range(episodes):
        ET = 0
        state = env.reset()
        action = policy(state)
        state_action_reward = [(state, action, None)]
        for _ in range(max_steps):
            state, reward, done, _ = env.step(action)
            action = policy(state)
            state_action_reward.append((state, action, reward))
            if done:
                break
        T = len(state_action_reward) - 1
        for a in range(T):
            state, _, _ = state_action_reward[a]
            state_t_1, _, reward_t_1 = state_action_reward[a + 1]
            ET *= lambtha * gamma
            ET += 1
            delta = reward_t_1 + gamma * V[state_t_1] - V[state]
            V[state] += alpha * delta * ET
    return V
