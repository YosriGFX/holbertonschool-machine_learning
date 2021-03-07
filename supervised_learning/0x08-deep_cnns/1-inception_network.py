#!/usr/bin/env python3
''' Inception Network'''
import tensorflow.keras as K

inception_block = __import__('0-inception_block').inception_block


def inception_network():
    '''Function that builds the inception
    network as described in Going Deeper
    with Convolutions (2014)'''
    MaxPooling2D = K.layers.MaxPooling2D
    AveragePooling2D = K.layers.AveragePooling2D
    Conv2D = K.layers.Conv2D
    Dropout = K.layers.Dropout(rate=0.4)
    Dense = K.layers.Dense
    X = K.Input(
        shape=(224, 224, 3)
    )
    layer7x7 = Conv2D(
        64,
        7,
        2,
        padding='same',
        activation='relu'
    )(X)
    layerMax = MaxPooling2D(
        3, 2, padding='same'
    )(layer7x7)
    layer3x3 = Conv2D(
        64, 1, 1, activation='relu'
    )(layerMax)
    layer3x3 = Conv2D(
        192,
        3,
        1,
        padding='same',
        activation='relu'
    )(layer3x3)
    layerMax_2 = MaxPooling2D(
        3, 2, padding='same'
    )(layer3x3)
    inception = inception_block(
        layerMax_2,
        [64, 96, 128, 16, 32, 32]
    )
    inception = inception_block(
        inception,
        [128, 128, 192, 32, 96, 64]
    )
    layerMax_2 = MaxPooling2D(
        3, 2, padding='same'
    )(inception)
    inception = inception_block(
        layerMax_2, [192, 96, 208, 16, 48, 64]
    )
    inception = inception_block(
        inception, [160, 112, 224, 24, 64, 64]
    )
    inception = inception_block(
        inception, [128, 128, 256, 24, 64, 64]
    )
    inception = inception_block(
        inception, [112, 144, 288, 32, 64, 64]
    )
    inception = inception_block(
        inception, [256, 160, 320, 32, 128, 128]
    )
    layerMax_2 = MaxPooling2D(
        3, 2, padding='same'
    )(inception)
    inception = inception_block(
        layerMax_2, [256, 160, 320, 32, 128, 128]
    )
    inception = inception_block(
        inception, [384, 192, 384, 48, 128, 128]
    )
    layerAVG_2 = AveragePooling2D(
        7, 1
    )(inception)
    dropped = Dropout(layerAVG_2)
    Y = Dense(
        1000, activation="softmax"
    )(dropped)
    out = K.Model(inputs=X, outputs=Y)
    return out
