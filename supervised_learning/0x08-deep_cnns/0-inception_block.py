#!/usr/bin/env python3
'''Inception Block'''
import tensorflow.keras as K


def inception_block(A_prev, filters):
    '''A Function that builds an inception
    block as described in Going Deeper
    with Convolutions (2014)'''
