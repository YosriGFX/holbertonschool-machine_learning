#!/usr/bin/env python3
'''6. Transformer Encoder Block'''
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class EncoderBlock(tf.keras.layers.Layer):
    '''create an encoder block for a transformer'''
    def __init__(self, dm, h, hidden, drop_rate=0.1):
        '''Class constructor'''
        super(EncoderBlock, self).__init__()
        self.mha = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(
            hidden,
            activation='relu'
        )
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6
        )
        self.layernorm2 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6
        )

        self.dropout1 = tf.keras.layers.Dropout(
            drop_rate
        )
        self.dropout2 = tf.keras.layers.Dropout(
            drop_rate
        )

    def call(self, x, training, mask=None):
        '''Public instance method'''
        attention = self.mha(x, x, x, mask)[0]
        attention = self.dropout1(
            attention,
            training=training
        )
        first = self.layernorm1(x + attention)
        output = self.layernorm2(
            first + self.dropout2(
                self.dense_output(
                    self.dense_hidden(first)
                ), training=training
            )
        )
        return output
