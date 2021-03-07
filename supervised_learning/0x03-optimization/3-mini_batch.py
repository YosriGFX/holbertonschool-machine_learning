#!/usr/bin/env python3
'''Mini-Batch'''


def train_mini_batch(
    X_train,
    Y_train,
    X_valid,
    Y_valid,
    batch_size=32,
    epochs=5,
    load_path="/tmp/model.ckpt",
    save_path="/tmp/model.ckpt"
):
    '''Function that that trains
    a loaded neural network
    model using mini-batch
    gradient descent'''
