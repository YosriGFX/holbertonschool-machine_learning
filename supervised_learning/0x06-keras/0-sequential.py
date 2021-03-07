#!/usr/bin/env python3
'''Sequential File'''
import tensorflow.keras as K


def build_model(
    nx,
    layers,
    activations,
    lambtha,
    keep_prob
):
    '''Function that builds a neural
    network with the Keras library'''
    baseModel = K.Sequential()
    for i in range(len(layers)):
        layer = K.layers.Dense(
            layers[i],
            input_dim=nx,
            activation=activations[i],
            kernel_initializer=K.initializers.VarianceScaling(
                mode="fan_avg"
            ),
            kernel_regularizer=K.regularizers.l2(
                lambtha
            )
        )
        baseModel.add(layer)
        if i != (len(layers) - 1):
            dropped = K.layers.Dropout(
                rate=1 - keep_prob
            )
            baseModel.add(dropped)
    return baseModel
