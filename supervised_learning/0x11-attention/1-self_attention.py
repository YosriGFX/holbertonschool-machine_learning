#!/usr/bin/env python3
'''1. Self Attention'''
import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    '''calculate the attention for machine
    translation based on PDF'''
    def __init__(self, units):
        '''Class constructor'''
        super(SelfAttention, self).__init__()
        self.units = units
        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        '''Public instance method'''
        weights = tf.nn.softmax(
            self.V(
                tf.nn.tanh(
                    self.W(
                        tf.expand_dims(s_prev, 1)
                    ) + self.U(hidden_states)
                )
            ),
            axis=1
        )
        context = tf.reduce_sum(
            weights * hidden_states,
            axis=1
        )
        return context, weights
