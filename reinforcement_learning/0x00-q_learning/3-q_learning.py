#!/usr/bin/env python3
'''3. Q-learning'''
import numpy as np
epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(
    env,
    Q,
    episodes=5000,
    max_steps=100,
    alpha=0.1,
    gamma=0.99,
    epsilon=1,
    min_epsilon=0.1,
    epsilon_decay=0.05
):
    '''A function that performs Q-learning'''
    total_rewards = []
    old_epsilon = epsilon
    for episode in range(episodes):
        state = env.reset()
        rewards = 0
        for step in range(max_steps):
            action = epsilon_greedy(Q, state, epsilon)
            new_state, reward, done, info = env.step(action)
            if done and reward == 0:
                reward = -1
            next_state = np.max(Q[new_state])
            Q[state, action] = (1 - alpha) * Q[state, action] + (
                alpha * (
                    reward + gamma * next_state
                )
            )
            rewards += reward
            state = new_state
            if done:
                break
        epsilon = min_epsilon + (
            old_epsilon - min_epsilon
        ) * np.exp(
            -epsilon_decay * episode
        )
        total_rewards.append(rewards)
    return Q, total_rewards
