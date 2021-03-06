#!/usr/bin/env python3
'''Normal'''


class Normal():
    '''Represents a normal distribution'''

    def __init__(self, data=None, mean=0., stddev=1.):
        '''Class constructor'''
        if data is None:
            if stddev <= 0:
                raise ValueError('stddev must be a positive value')
            self.stddev = stddev
            self.mean = mean
        else:
            if type(data) is not list:
                raise TypeError('data must be a list')
            elif len(data) <= 2:
                raise ValueError('data must contain multiple values')
            else:
                self.mean = sum(data)/len(data)
                sumT = 0
                for x in data:
                    sumT += (x - self.mean)**2
                self.stddev = (sumT/len(data))**0.5

    def z_score(self, x):
        '''Calculates the z-score
        of a given x-value'''
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        '''Calculates the x-value
        of a given z-score'''
        return z * self.stddev + self.mean

    def pdf(self, x):
        '''Calculates the value of the PDF
        for a given x-value'''
        puiss = -(
            (
                x-self.mean
            ) ** 2 / (
                2*self.stddev**2
            )
        )
        return 2.7182818285 ** puiss / (
            self.stddev * (2 * 3.1415926536) ** 0.5
        )

    def cdf(self, x):
        '''Calculates the value of the CDF
        for a given x-value'''
        a = self.mean

        def erf(a):
            '''catch error'''
            an = (
                a - (
                    (a**3) / 3
                ) + ((a**5) / 10) - (
                    (a**7) / 42
                ) + ((a**9) / 216)
            )
            return 2 * an / (3.1415926536**0.5)
        n = ((x - a) / self.stddev)/(2**0.5)
        return (1 + erf(n))/2
