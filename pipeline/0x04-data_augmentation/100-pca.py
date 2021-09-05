#!/usr/bin/env python3
'''7. PCA Color Augmentation'''
import tensorflow as tf
import numpy as np


def pca_color(image, alphas):
    '''A function that performs PCA color augmentation
    as described in the AlexNet paper'''
    img = tf.keras.preprocessing.image.img_to_array(image)
    img_r = np.reshape(img, (img.shape[0] * img.shape[1], 3))
    mean = np.mean(img_r, axis=0)
    img_c = img_r - mean
    std = np.std(img_c, axis=0)
    img_c /= std
    cov = np.cov(img_c, rowvar=False)
    eig, p = np.linalg.eig(cov)
    delta = np.dot(p, alphas*eig)
    pca_a = img_c + delta
    pca = pca_a * std + mean
    pca = np.maximum(np.minimum(pca, 255), 0).astype('uint8')
    pca = pca.reshape((img.shape[0], img.shape[1], 3))
    return pca
