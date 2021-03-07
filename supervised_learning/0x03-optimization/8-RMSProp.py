#!/usr/bin/env python3
'''RMSProp Upgraded'''
import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    '''Function  that creates the
    training operation for a neural
    network in tensorflow using the
    RMSProp optimization algorithm'''
    return tf.train.RMSPropOptimizer(
        alpha,
        beta2,
        epsilon=epsilon
    ).minimize(loss)
