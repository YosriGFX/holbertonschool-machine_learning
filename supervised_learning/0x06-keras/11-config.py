#!/usr/bin/env python3
""" doc """
import tensorflow.keras as K


def save_config(network, filename):
    """ doc """
    with open(filename, "w+") as f:
        f.write(network.to_json())


def load_config(filename):
    """ doc """
    with open(filename, "r") as f:
        return(K.models.model_from_json(f.read()))
