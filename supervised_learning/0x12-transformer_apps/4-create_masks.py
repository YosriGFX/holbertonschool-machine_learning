#!/usr/bin/env python3
'''4. Create Masks'''
import tensorflow as tf


def create_masks(inputs, target):
    '''creates all masks for training/validation'''
    size = tf.shape(target)[1]
    encoder_mask = tf.cast(
        tf.math.equal(
            inputs, 0
        ),
        tf.float32
    )[
        :,
        tf.newaxis,
        tf.newaxis,
        :
    ]
    decoder_mask = tf.cast(
        tf.math.equal(
            inputs, 0
        ),
        tf.float32
    )[
        :,
        tf.newaxis,
        tf.newaxis,
        :
    ]
    combined_mask = tf.maximum(
        tf.cast(
            tf.math.equal(
                target, 0
            ),
            tf.float32
        )[
            :,
            tf.newaxis,
            tf.newaxis,
            :
        ],
        1 - tf.linalg.band_part(
            tf.ones(
                (
                    size,
                    size
                )
            ),
            -1,
            0
        )
    )
    return encoder_mask, combined_mask, decoder_mask
