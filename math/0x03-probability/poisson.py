#!/usr/bin/env python3
'''Poisson'''


class Poisson():
    '''class Poisson represents
    a poisson distribution'''

    def __init__(self, data=None, lambtha=1.):
        '''Class constructor'''
        if data is None:
            if lambtha <= 0:
                raise ValueError('lambtha must be a positive value')
            else:
                self.lambtha = float(lambtha)
        else:
            if type(data) is not list:
                raise TypeError('data must be a list')
            elif len(data) < 2:
                raise ValueError('data must contain multiple values')
            else:
                self.lambtha = sum(data)/len(data)
