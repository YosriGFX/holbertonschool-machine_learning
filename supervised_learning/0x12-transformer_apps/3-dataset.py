#!/usr/bin/env python3
'''3. Dataset'''
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset:
    '''loads and preps a dataset for machine translation'''
    def __init__(self, batch_size, max_len):
        '''Class constructor'''
        self.data_train, self.data_valid = tfds.load(
            "ted_hrlr_translate/pt_to_en",
            split=['train', 'validation'],
            as_supervised=True
        )
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )
        self.data_train = data_train.map(self.tf_encode)
        self.data_train.filter(
            lambda x, y: tf.logical_and(
                tf.size(x) <= max_len,
                tf.size(y) <= max_len
            )
        )
        self.data_train = self.data_train.cache()
        data_size = sum(
            1 for __ in self.data_train
        )
        self.data_train = self.data_train.shuffle(
            data_size
        )
        self.data_train = self.data_train.padded_batch(
            batch_size
        )
        self.data_train = self.data_train.prefetch(
            tf.data.experimental.AUTOTUNE
        )

        self.data_valid = data_valid.map(self.tf_encode)
        self.data_valid.filter(
            lambda x, y: tf.logical_and(
                tf.size(x) <= max_len,
                tf.size(y) <= max_len
            )
        )
        self.data_valid = self.data_valid.padded_batch(
            batch_size
        )

    def tokenize_dataset(self, data):
        '''creates sub-word tokenizers for our dataset'''
        Subword = tfds.deprecated.text.SubwordTextEncoder
        tokenizer_pt = Subword.build_from_corpus(
            (
                a.numpy() for a, __ in data
            ),
            target_vocab_size=2**15
        )
        tokenizer_en = Subword.build_from_corpus(
            (
                b.numpy() for __, b in data
            ),
            target_vocab_size=2 ** 15
        )
        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        '''encodes a translation into tokens'''
        pt_vsize = self.tokenizer_pt.vocab_size
        pt_tokens = [
            pt_vsize
        ] + self.tokenizer_pt.encode(
            pt.numpy()
        ) + [
            pt_vsize + 1
        ]
        en_tokens = [
            self.tokenizer_en.vocab_size
        ] + self.tokenizer_en.encode(
            en.numpy()
        ) + [
            pt_vsize + 1
        ]

        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        '''acts as a tensorflow wrapper for the encode instance method'''
        pt_lang, en_lang = tf.py_function(
            func=self.encode,
            inp=[
                pt,
                en
            ],
            Tout=[
                tf.int64,
                tf.int64
            ]
        )
        pt_lang.set_shape([None])
        en_lang.set_shape([None])
        return pt_lang, en_lang
