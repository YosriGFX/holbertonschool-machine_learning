#!/usr/bin/env python3
'''L2 Reg Create Layer'''
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    '''Function that creates a tensorflow
    layer that includes L2 regularization'''
    layer = tf.layers.Dense(
        n,
        activation,
        name='layer',
        kernel_initializer=(
            tf.contrib.layers.variance_scaling_initializer(
                mode="FAN_AVG"
            )
        ),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(
            lambtha
        )
    )(prev)
    return layer
