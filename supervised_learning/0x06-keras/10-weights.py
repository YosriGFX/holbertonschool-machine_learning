#!/usr/bin/env python3
'''Save and Load Weights'''
import tensorflow.keras as K


def save_weights(network, filename, save_format='h5'):
    '''Function that saves a model’s weights'''
    if filename[-2:] != save_format:
        filename += save_format
    network.save_weights(filename)


def load_weights(network, filename):
    '''Function that loads a model’s weights'''
    network.load_weights(filename)
