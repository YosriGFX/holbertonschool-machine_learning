#!/usr/bin/env python3
'''Binominal'''


class Binomial():
    '''Represents a Binomial distribution'''

    def __init__(self, data=None, n=1, p=0.5):
        '''Class Constructor'''
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            elif (
                p <= 0
            ) or (
                p >= 1
            ):
                raise ValueError("p must be greater than 0 and less than 1")
            else:
                self.p = float(p)
                self.n = int(n)
        else:
            if type(data) is not list:
                raise TypeError('data must be a list')
            elif len(data) <= 2:
                raise ValueError('data must contain multiple values')
            else:
                meanObj = sum(data)/len(data)
                s = 0
                for i in data:
                    s += (i - meanObj)**2
                n = round((meanObj**2) / (meanObj - s / len(data)))
                self.n = int(n)
                self.p = meanObj / n

    def fact(self, k):
        '''Calculate factorial'''
        fact = 1
        for i in range(1, k + 1):
            fact = fact * i
        return fact

    def pmf(self, k):
        '''Calculates the value of the PMF
        for a given number of “successes”'''
        if k > 0:
            try:
                k = int(k)
                nk = self.fact(
                    self.n
                ) / (
                    self.fact(
                        k
                    ) * self.fact(
                        self.n - k
                    )
                )
                return nk * self.p**k * (
                    1-self.p
                ) ** (
                    self.n-k
                )
            except Exception:
                return 0
        else:
            return 0

    def cdf(self, k):
        '''Calculates the value of the CDF
        for a given number of “successes”'''
        if k > 0:
            try:
                k = int(k)
                cdf = 0
                for i in range(0, k + 1):
                    cdf += self.pmf(i)
                return cdf
            except Exception:
                return 0
        else:
            return 0
