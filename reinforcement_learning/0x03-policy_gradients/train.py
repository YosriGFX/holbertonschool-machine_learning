#!/usr/bin/env python3
'''3. Animate iteration'''
import numpy as np
from policy_gradient import policy
from policy_gradient import policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98, show_result=False):
    '''A function that implements a full training.'''
    def play_episode(env, weight, episode, show_result):
        '''A function that plays a single episode.'''
        state = env.reset()[None, :]
        state_action_reward_grad = []
        while True:
            if show_result and (episode % 1000 == 0):
                env.render()
            action, grad = policy_gradient(state, weight)
            state, reward, done, _ = env.step(action)
            state = state[None, :]
            state_action_reward_grad.append((state, action, reward, grad))
            if done:
                break
        env.close()
        return state_action_reward_grad

    weight = np.random.rand(4, 2)
    episodes = []
    for episode in range(nb_episodes):
        sarg = play_episode(env, weight, episode, show_result)
        T = len(sarg) - 1
        score = 0
        for t in range(0, T):
            _, _, reward, grad = sarg[t]
            score += reward

            G = np.sum([
                gamma**sarg[k][2] *
                sarg[k][2] for k in range(t + 1, T + 1)])
            weight += alpha * G * grad
        episodes.append(score)
        print("{}: {}".format(episode, score), end="\r", flush=False)
    return episodes
