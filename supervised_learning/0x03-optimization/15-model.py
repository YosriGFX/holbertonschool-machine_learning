#!/usr/bin/env python3
'''Put it all together and what do you get'''
import numpy as np
import tensorflow as tf


def model(
    Data_train,
    Data_valid,
    layers,
    activations,
    alpha=0.001,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
    decay_rate=1,
    batch_size=32,
    epochs=5,
    save_path='/tmp/model.ckpt'
):
    '''Function that builds, trains, and saves
    a neural network model in tensorflow using
    Adam optimization, mini-batch gradient descent,
    learning rate decay, and batch normalization'''
    return None
