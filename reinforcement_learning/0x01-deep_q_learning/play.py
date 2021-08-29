#!/usr/bin/env python3
'''A python script that can display a game
played by the agent trained by train.py'''
import gym
import h5py
import numpy as np
from rl.agents.dqn import DQNAgent
import keras as K
from keras.optimizers import Adam
from rl.memory import SequentialMemory
from rl.policy import GreedyQPolicy
train_agent = __import__('train').train_agent


env = gym.make('Breakout-v0')
train_agent(env)
state = env.reset()
DQN_agent = DQNAgent(
    model=K.models.load_model('policy.h5'),
    nb_actions=env.action_space.n,
    memory=SequentialMemory(
        limit=1000000,
        window_length=4
    ),
    policy=GreedyQPolicy()
)
DQN_agent.compile(
    optimizer=Adam(
        lr=.00025,
        clipnorm=1.0
    ),
    metrics=['mae']
)
DQN_agent.test(
    env,
    nb_episodes=10,
    visualize=True
)
