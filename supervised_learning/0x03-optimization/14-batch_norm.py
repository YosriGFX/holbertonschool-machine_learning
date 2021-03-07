#!/usr/bin/env python3
'''Batch Normalization Upgraded'''
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    '''Function that creates a batch
    normalization layer for a neural
    network in tensorflow'''
    initiator = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG"
    )
    layer = tf.layers.Dense(
        n,
        kernel_initializer=initiator
    )(prev)
    m, v = tf.nn.moments(
      layer, axes=[0]
    )
    beta = tf.Variable(
      tf.zeros([n])
    )
    gamma = tf.Variable(
      tf.ones([n])
    )
    eps = 1e-8,
    scale = None
    out = tf.nn.batch_normalization(
      layer, m, v, beta, gamma, eps, scale
    )
    return activation(out)
