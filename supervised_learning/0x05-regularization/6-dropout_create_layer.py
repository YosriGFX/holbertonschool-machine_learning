#!/usr/bin/env python3
'''Create a Layer with Dropout'''
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    '''A Function that creates a layer
    of a neural network using dropout'''
    layer = tf.layers.Dense(
        n,
        activation,
        name='layer',
        kernel_initializer=(
            tf.contrib.layers.variance_scaling_initializer(
                mode="FAN_AVG"
            )
        ),
        activity_regularizer=tf.layers.Dropout(rate=keep_prob)
    )(prev)
    return (layer)
