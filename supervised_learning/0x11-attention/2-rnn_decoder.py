#!/usr/bin/env python3
'''2. RNN Decoder'''
import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    '''decode for machine translation'''
    def __init__(self, vocab, embedding, units, batch):
        '''Class constructor'''
        super(RNNDecoder, self).__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(
            vocab,
            embedding
        )
        self.gru = tf.keras.layers.GRU(
            self.units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer="glorot_uniform"
        )
        self.F = tf.keras.layers.Dense(
            vocab
        )

    def call(self, x, s_prev, hidden_states):
        '''Public instance method'''
        Self_Attention = SelfAttention(self.units)
        context = Self_Attention(
            s_prev, hidden_states
        )[0]
        x = self.embedding(x)
        context = tf.expand_dims(context, 1)
        output, state = self.gru(
            inputs=tf.concat(
                [context, x],
                axis=-1
            )
        )
        output = tf.reshape(
            output,
            (
                -1, output.shape[2]
            )
        )
        y = self.F(output)
        return y, state
