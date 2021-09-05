#!/usr/bin/env python3
'''2. Rotate'''
import tensorflow as tf


def rotate_image(image):
    '''A function that rotates an image
    by 90 degrees counter-clockwis'''
    return tf.image.rot90(
        image,
        k=1
    )
