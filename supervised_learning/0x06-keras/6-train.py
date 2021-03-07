#!/usr/bin/env python3
'''Early Stopping'''
import tensorflow.keras as K


def train_model(
    network,
    data,
    labels,
    batch_size,
    epochs,
    validation_data=None,
    early_stopping=False,
    patience=0,
    verbose=True,
    shuffle=False
):
    '''Function That trains a model
    using mini-batch gradient descent
    to also analyze validaiton data'''
    callbacks = None
    if validation_data:
        if early_stopping:
            callbacks = [K.callbacks.EarlyStopping(
                monitor="loss",
                patience=patience,
                mode="auto"
            )]
    return network.fit(
        data,
        labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        callbacks=callbacks,
        validation_data=validation_data,
        shuffle=shuffle
    )
