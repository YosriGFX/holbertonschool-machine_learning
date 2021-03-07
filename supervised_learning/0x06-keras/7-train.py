#!/usr/bin/env python3
'''Learning Rate Decay'''
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
    learning_rate_decay=False,
    alpha=0.1,
    decay_rate=1,
    verbose=True,
    shuffle=False
):
    '''Function to also train the
    model with learning rate decay'''
    callbacks = []
    if validation_data:
        if early_stopping:
            callbacks = [K.callbacks.EarlyStopping(
                monitor="loss",
                patience=patience,
                mode="auto"
            )]

        if learning_rate_decay:
            def scheduler(epoch):
                '''to also train the model'''
                return alpha / (
                    1 + (
                        decay_rate * (epoch)
                    )
                )
            callbacks.append(
                K.callbacks.LearningRateScheduler(
                    scheduler, 1
                )
            )
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
