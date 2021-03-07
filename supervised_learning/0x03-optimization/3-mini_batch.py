#!/usr/bin/env python3
'''Mini-Batch'''
import tensorflow as tf

shuffle_data = __import__('2-shuffle_data').shuffle_data


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
    return None
