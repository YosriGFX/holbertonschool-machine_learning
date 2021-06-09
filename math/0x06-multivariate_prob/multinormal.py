#!/usr/bin/env python3
'''Mean and Covariance'''
import numpy as np


class MultiNormal():
    '''Multivariate Normal distribution'''

    def __init__(self, data):
        '''class constructor'''
        if type(data) is np.ndarray and len(data.shape) == 2:
            if data.shape[1] > 2:
                a, b = data.shape
                self.mean = (
                    np.mean(data, axis=1)
                ).reshape(a, 1)
                X = data - self.mean
                self.cov = (
                    (
                        np.matmul(X, X.T)
                    ) / (b - 1)
                )
            else:
                raise ValueError('data must contain multiple data points')
        else:
            raise TypeError('data must be a 2D numpy.ndarray')

    def pdf(self, x):
        '''Calculates the PDF at a data point'''
        if isinstance(x, np.ndarray):
            dimension = self.cov.shape[0]
            if len(x.shape) == 2\
                and x.shape[1] == 1\
                    and x.shape[0] == dimension:
                cov_det = np.linalg.det(self.cov)
                cov_in = np.linalg.inv(self.cov)
                x_u = x - self.mean
                first_pdf = 1 / np.sqrt(
                    (
                        (2 * np.pi) ** dimension
                    ) * cov_det
                )
                second_pdf = np.exp(
                    np.matmul(
                        np.matmul(
                            -x_u.T / 2, cov_in
                        ),
                        x_u
                    )
                )
                pdf = first_pdf * second_pdf
                return pdf.flatten()[0]
            else:
                raise ValueError(
                    'x must have the shape ({}, 1)'.format(dimension)
                )
        else:
            raise TypeError('x must be a numpy.ndarray')
