#!/usr/bin/env python3
'''5. Multi Head Attention'''
import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    '''perform multi head attention'''
    def __init__(self, dm, h) -> None:
        '''Class constructor'''
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.dm = dm
        self.depth = self.dm // self.h
        self.Wq = tf.keras.layers.Dense(self.dm)
        self.Wk = tf.keras.layers.Dense(self.dm)
        self.Wv = tf.keras.layers.Dense(self.dm)
        self.linear = tf.keras.layers.Dense(self.dm)

    def call(self, Q, K, V, mask):
        '''Public instance method'''
        size = tf.shape(Q)[0]
        q = self.Wq(Q)
        k = self.Wk(K)
        v = self.Wv(V)
        q = tf.transpose(
            tf.reshape(
                q,
                (
                    size,
                    -1,
                    self.h,
                    self.depth
                )
            ), perm=[0, 2, 1, 3]
        )
        k = tf.transpose(
            tf.reshape(
                k,
                (
                    size,
                    -1,
                    self.h,
                    self.depth
                )
            ), perm=[0, 2, 1, 3]
        )
        v = tf.transpose(
            tf.reshape(
                v,
                (
                    size,
                    -1,
                    self.h,
                    self.depth
                )
            ), perm=[0, 2, 1, 3]
        )
        output, weights = sdp_attention(q, k, v, mask)
        output = self.linear(
            tf.reshape(
                tf.transpose(
                    output, perm=[0, 2, 1, 3]
                ),
                (
                    size, -1, self.dm
                )
            )
        )
        return output, weights
