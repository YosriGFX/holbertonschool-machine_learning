#!/usr/bin/env python3
'''8. Transformer Encoder'''
import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
EncoderBlock = __import__('7-transformer_encoder_block').EncoderBlock


class Encoder(tf.keras.layers.Layer):
    '''encoder for a transformer'''
    def __init__(
        self,
        N,
        dm,
        h,
        hidden,
        input_vocab,
        max_seq_len,
        drop_rate=0.1
    ):
        '''Class constructor'''
        super(Encoder, self).__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(
            input_vocab,
            dm
        )
        self.positional_encoding = positional_encoding(
            max_seq_len,
            dm
        )
        self.blocks = [
            EncoderBlock(
                dm,
                h,
                hidden,
                drop_rate
            ) for __ in range(self.N)
        ]
        self.dropout = tf.keras.layers.Dropout(
            drop_rate
        )

    def call(self, x, training, mask):
        '''Public instance method'''
        x = self.dropout(
            self.embedding(x) * tf.math.sqrt(
                tf.cast(self.dm, tf.float32)
            ) + self.positional_encoding[:x.shape[1], :],
            training=training
        )
        for block in self.blocks:
            x = block(x, training, mask)
        return x
