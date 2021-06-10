#!/usr/bin/env python3
'''10. Transformer Network'''
import tensorflow as tf


def positional_encoding(max_seq_len, dm):
    '''calculates the positional encoding for a transformer'''
    pos = np.arange(
        max_seq_len
    )
    PE = pos[:, np.newaxis] * 1 / np.power(
        10000, (
            2 * (
                np.arange(
                    dm
                )[np.newaxis, :] // 2
            )
        ) / np.float32(dm)
    )
    PE[:, 0::2] = np.sin(PE[:, 0::2])
    PE[:, 1::2] = np.cos(PE[:, 1::2])
    return PE


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


class Decoder(tf.keras.layers.Layer):
    '''Decoder for a transformer'''
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
        super(Decoder, self).__init__()
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
            DecoderBlock(
                dm,
                h,
                hidden,
                drop_rate
            ) for __ in range(self.N)
        ]
        self.dropout = tf.keras.layers.Dropout(
            drop_rate
        )

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        '''Public instance method'''
        x = self.dropout(
            self.embedding(x) * tf.math.sqrt(
                tf.cast(self.dm, tf.float32)
            ) + self.positional_encoding[:x.shape[1], :],
            training=training
        )
        for block in self.blocks:
            x = block(
                x,
                encoder_output,
                training,
                look_ahead_mask,
                padding_mask
            )
        return x


class Transformer(tf.keras.Model):
    '''create a transformer network'''
    def __init__(
        self,
        N,
        dm,
        h,
        hidden,
        input_vocab,
        target_vocab,
        max_seq_input,
        max_seq_target,
        drop_rate=0.1
    ):
        '''Class constructor'''
        super(Transformer, self).__init__()
        self.encoder = Encoder(
            N,
            dm,
            h,
            hidden,
            input_vocab,
            max_seq_input,
            drop_rate
        )
        self.decoder = Decoder(
            N,
            dm,
            h,
            hidden,
            target_vocab,
            max_seq_target,
            drop_rate
        )
        self.linear = tf.keras.layers.Dense(
            target_vocab
        )

    def call(
        self,
        inputs,
        target,
        training,
        encoder_mask,
        look_ahead_mask,
        decoder_mask
    ):
        '''Public instance method'''
        output = self.linear(
            self.decoder(
                target,
                self.encoder(
                    inputs,
                    training,
                    encoder_mask
                ),
                training,
                look_ahead_mask,
                decoder_mask
            )
        )
        return output
