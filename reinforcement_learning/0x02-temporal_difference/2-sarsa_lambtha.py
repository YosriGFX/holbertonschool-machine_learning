#!/usr/bin/env python3
'''2. SARSA(λ)'''
import numpy as np


def sarsa_lambtha(
    env,
    Q,
    lambtha,
    episodes=5000,
    max_steps=100,
    alpha=0.1,
    gamma=0.99,
    epsilon=1,
    min_epsilon=0.1,
    epsilon_decay=0.05
):
    '''A function that performs SARSA(λ)'''
    policy = np.zeros(shape=Q.shape) + epsilon
    maxq = np.argmax(Q, axis=-1)
    policy[range(len(policy)), maxq] = 1 - epsilon

    for episode in range(episodes):
        ET = 0
        state = env.reset()
        action = int(np.random.choice(policy[state]))
        state_action_reward = [(state, action, None)]
        for _ in range(max_steps):
            state, reward, done, _ = env.step(action)
            action = int(np.random.choice(policy[state]))
            state_action_reward.append((state, action, reward))
            if done:
                break
        T = len(state_action_reward) - 1
        for t in range(T):
            state, action, _ = state_action_reward[t]
            state_t_1, action_t_1, reward_t_1 = state_action_reward[t + 1]
            ET *= lambtha * gamma
            ET += 1
            delta = reward_t_1 + gamma * Q[state_t_1, action_t_1] - \
                Q[state, action]

            Q += alpha * delta * ET
        epsilon = min_epsilon + \
            (epsilon - min_epsilon) * np.exp(-epsilon_decay * episode)
        policy = np.zeros(shape=Q.shape) + epsilon
        maxq = np.argmax(Q, axis=-1)
        policy[range(len(policy)), maxq] = 1 - epsilon
    return Q
