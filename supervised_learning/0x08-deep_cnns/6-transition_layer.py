#!/usr/bin/env python3
'''Transition Layer'''
import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    '''Functionn that builds a transition
    layer as described in Densely
    Connected Convolutional Networks:'''
    MaxPooling2D = K.layers.MaxPooling2D
    AveragePooling2D = K.layers.AveragePooling2D
    Conv2D = K.layers.Conv2D
    Dense = K.layers.Dense
    BatchNorm = K.layers.BatchNormalization
    Activation = K.layers.Activation
    Concatenate = K.layers.Concatenate
    newLayer = BatchNorm()(X)
    newLayer = Activation('relu')(newLayer)
    nb_filters = int(nb_filters * compression)
    newLayer = Conv2D(
        nb_filters,
        1,
        1,
        padding='valid',
        kernel_initializer='he_normal'
    )(newLayer)
    newLayer = AveragePooling2D(2)(newLayer)
    return newLayer, nb_filters
