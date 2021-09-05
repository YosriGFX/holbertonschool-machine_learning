#!/usr/bin/env python3
'''0. Flip'''
import tensorflow as tf


def flip_image(image):
    '''A function that flips
    an image horizontally'''
    return tf.image.flip_left_right(image)
