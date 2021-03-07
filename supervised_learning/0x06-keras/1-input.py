#!/usr/bin/env python3
'''Input File'''
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
    inputs = K.Input(shape=(nx,))
    x = inputs
    for i in range(0, len(layers)):
        dropped = K.layers.Dropout(
            rate=1 - keep_prob
        )
        x = K.layers.Dense(
            layers[i],
            activation=activations[i],
            kernel_initializer=K.initializers.VarianceScaling(
                mode="fan_avg"
            ),
            kernel_regularizer=K.regularizers.l2(lambtha)
        )(x)
        if i != len(layers)-1:
            x = dropped(x)
    baseModel = K.Model(inputs=inputs, outputs=x)
    return baseModel
