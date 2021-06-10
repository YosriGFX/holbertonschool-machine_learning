#!/usr/bin/env python3
'''0. RNN Encoder'''
import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    '''encode for machine translation'''
    def __init__(self, vocab, embedding, units, batch):
        '''Class constructor'''
        super(RNNEncoder, self).__init__()
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

    def initialize_hidden_state(self):
        '''Initializes the hidden states for
        the RNN cell to a tensor of zeros'''
        return tf.zeros(
            shape=(
                self.batch,
                self.units
            )
        )

    def call(self, x, initial):
        '''Public instance method'''
        return self.gru(
            inputs=self.embedding(x),
            initial_state=initial
        )
