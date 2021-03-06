#!/usr/bin/env python3
'''Create Layer'''
import tensorflow as tf


def create_layer(prev, n, activation):
    '''Function that Crreate Layer'''
    layer = tf.layers.Dense(
        n,
        activation,
        name='layer',
        kernel_initializer=(
            tf.contrib.layers.variance_scaling_initializer(
                mode="FAN_AVG"
            )
        )
    )(prev)
    return layer
