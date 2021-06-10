#!/usr/bin/env python3
'''7. Transformer Decoder Block'''
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class DecoderBlock(tf.keras.layers.Layer):
    '''create an encoder block for a transformer'''
    def __init__(self, dm, h, hidden, drop_rate=0.1):
        '''Class constructor'''
        super(DecoderBlock, self).__init__()
        self.mha1 = MultiHeadAttention(
            dm, h
        )
        self.mha2 = MultiHeadAttention(
            dm, h
        )
        self.dense_hidden = tf.keras.layers.Dense(
            hidden,
            activation="relu"
        )
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6
        )
        self.layernorm2 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6
        )
        self.layernorm3 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6
        )
        self.dropout1 = tf.keras.layers.Dropout(
            drop_rate
        )
        self.dropout2 = tf.keras.layers.Dropout(
            drop_rate
        )
        self.dropout3 = tf.keras.layers.Dropout(
            drop_rate
        )

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        '''Public instance method'''
        First = self.dropout1(
            self.mha1(
                x, x, x, look_ahead_mask
            )[0],
            training=training
        )
        FirstOUT = self.layernorm1(x + First)
        Second = self.dropout2(
            self.mha2(
                FirstOUT,
                encoder_output,
                encoder_output,
                padding_mask
            )[0],
            training=training
        )
        SecondOUT = self.layernorm2(Second + FirstOUT)
        ThirdOUT = self.dropout3(
            self.dense_output(
                self.dense_hidden(
                    SecondOUT
                )
            ),
            training=training
        )
        output = self.layernorm3(
            ThirdOUT + SecondOUT
        )
        return output
