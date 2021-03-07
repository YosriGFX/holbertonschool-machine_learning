#!/usr/bin/env python3
'''Testing File'''
import tensorflow.keras as K


def test_model(
    network, data, labels, verbose=True
):
    '''A Function that tests
    a neural network'''
    return network.evaluate(
        x=data,
        y=labels,
        verbose=verbose
    )
