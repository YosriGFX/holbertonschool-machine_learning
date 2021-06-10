#!/usr/bin/env python3
'''4. Scaled Dot Product Attention'''
import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    '''calculates the a dot product attention'''
    a = tf.matmul(
        Q, K, transpose_b=True
    ) / tf.sqrt(
        tf.cast(
            tf.shape(K)[-1],
            tf.float32
        )
    )
    if mask:
        a += mask
    weights = tf.nn.softmax(a, axis=-1)
    output = tf.matmul(weights, V)
    return output, weights
