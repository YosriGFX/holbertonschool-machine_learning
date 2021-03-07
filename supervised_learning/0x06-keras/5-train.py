#!/usr/bin/env python3
'''Train'''
import tensorflow.keras as K


def train_model(
    network,
    data,
    labels,
    batch_size,
    epochs,
    validation_data=None,
    verbose=True,
    shuffle=False
):
    '''Function That trains a model
    using mini-batch gradient descent
    to also analyze validaiton data'''
    return network.fit(
        data,
        labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        validation_data=validation_data,
        shuffle=shuffle
    )
