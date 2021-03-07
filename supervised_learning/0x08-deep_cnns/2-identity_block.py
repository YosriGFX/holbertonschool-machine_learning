#!/usr/bin/env python3
'''Identity Block'''
import tensorflow.keras as K


def identity_block(A_prev, filters):
    '''A Function that builds an identity
    block as described in Deep Residual
    Learning for Image Recognition (2015)'''
    Conv2D = K.layers.Conv2D
    BatchNorm = K.layers.BatchNormalization
    Activation = K.layers.Activation
    Add = K.layers.Add
    F11, F3, F12 = filters
    layer1x1 = Conv2D(
        F11,
        1,
        kernel_initializer='he_normal'
    )(A_prev)
    layer1x1 = BatchNorm()(layer1x1)
    layer1x1 = Activation('relu')(layer1x1)
    layer3x3 = Conv2D(
        F3,
        3,
        padding='same',
        kernel_initializer='he_normal'
    )(layer1x1)
    layer3x3 = BatchNorm()(layer3x3)
    layer3x3 = Activation('relu')(layer3x3)
    layer1x1 = Conv2D(
        F12,
        1,
        kernel_initializer='he_normal'
    )(layer3x3)
    layer1x1 = BatchNorm()(layer1x1)
    out = Add()([layer1x1, A_prev])
    out = Activation(
        'relu'
    )(out)
    return out
