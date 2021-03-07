#!/usr/bin/env python3
'''Inception Block'''
import tensorflow.keras as K


def inception_block(A_prev, filters):
    '''A Function that builds an inception
    block as described in Going Deeper
    with Convolutions (2014)'''
    MaxPooling2D = K.layers.MaxPooling2D
    Conv2D = K.layers.Conv2D
    Concatenate = K.layers.Concatenate()
    F1, F3R, F3, F5R, F5, FPP = filters
    layer1_0 = Conv2D(
        F1,
        1,
        activation='relu'
    )(A_prev)
    layer1_1 = Conv2D(
        F3R,
        1,
        padding='same',
        activation='relu'
    )(A_prev)
    layer3 = Conv2D(
        F3,
        3,
        padding='same',
        activation='relu'
    )(layer1_1)
    layer1_2 = Conv2D(
        F5R,
        1,
        padding='same',
        activation='relu'
    )(A_prev)
    layer5 = Conv2D(
        F5,
        5,
        padding='same',
        activation='relu'
    )(layer1_2)
    layerMax = MaxPooling2D(1)(A_prev)
    layer1_3 = Conv2D(
        FPP,
        1,
        padding='same',
        activation='relu'
    )(layerMax)
    out = Concatenate(
        [
            layer1_0,
            layer3,
            layer5,
            layer1_3
        ]
    )
    return out
