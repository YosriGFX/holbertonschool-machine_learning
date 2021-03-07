#!/usr/bin/env python3
'''Save and Load Model'''
import tensorflow.keras as K


def save_model(network, filename):
    '''Function that Save and Load Model'''
    network.save(
        filename
    )


def load_model(filename):
    '''Function that loads an entire model'''
    return K.models.load_model(
        filename
    )
