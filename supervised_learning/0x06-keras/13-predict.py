#!/usr/bin/env python3
'''Predict File'''
import tensorflow.keras as K


def predict(network, data, verbose=False):
    '''Function that makes a prediction
    using a neural network'''
    return network.predict(
        data,
        verbose=verbose
    )
