#!/usr/bin/env python3
'''Dense Block'''
import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    '''Function that builds a dense block
    as described in Densely Connected
    Convolutional Networks'''
    def Converser(X, k, f, val='valid', sc=1):
        '''Layer Converser'''
        layer = Conv2D(
            k,
            f,
            sC,
            padding=val,
            kernel_initializer='he_normal'
        )(X)
        return layer
    MaxPooling2D = K.layers.MaxPooling2D
    AveragePooling2D = K.layers.AveragePooling2D
    Conv2D = K.layers.Conv2D
    Dense = K.layers.Dense
    BatchNorm = K.layers.BatchNormalization
    Activation = K.layers.Activation
    Concatenate = K.layers.Concatenate
    layer_prev = X
    for layer in range(layers):
        layer_new = BatchNorm()(layer_prev)
        layer_new = Activation(
            'relu'
        )(layer_new)
        layer_new = Converser(
            layer_new, growth_rate * 4, 1, 'same'
        )
        layer_new = BatchNorm()(layer_new)
        layer_new = Activation(
            'relu'
        )(layer_new)
        layer_new = Converser(
            layer_new, growth_rate, 3, 'same'
        )
        layer_prev = Concatenate()(
            [layer_prev, layer_new]
        )
        nb_filters += growth_rate
    return layer_prev, nb_filters
