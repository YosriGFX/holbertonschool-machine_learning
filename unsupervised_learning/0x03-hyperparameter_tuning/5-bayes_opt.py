#!/usr/bin/env python3
'''5. Bayesian Optimization'''
import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    '''performs Bayesian optimization
    on a noiseless 1D Gaussian process'''
    def __init__(
        self,
        f,
        X_init,
        Y_init,
        bounds,
        ac_samples,
        l=1,
        sigma_f=1,
        xsi=0.01,
        minimize=True
    ):
        '''Class constructor'''
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        min, max = bounds
        self.X_s = np.linspace(min, max, ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        '''calculates the next best sample location'''
        mu, sigma = self.gp.predict(self.X_s)
        sigma = sigma.reshape(-1, 1)
        with np.errstate(
            divide='warn'
        ):
            if self.minimize is True:
                tmp = (
                    np.amin(self.gp.Y) - mu - self.xsi
                ).reshape(-1, 1)
            else:
                tmp = (
                    mu - np.amax(self.gp.Y) - self.xsi
                ).reshape(-1, 1)
            EI = tmp * norm.cdf(
                tmp / sigma
            ) + sigma * norm.pdf(
                tmp / sigma
            )
            EI[sigma == 0.0] = 0.0
            EI = EI.reshape(-1)
            X_next = self.X_s[np.argmax(EI)]
        return X_next, EI

    def optimize(self, iterations=100):
        '''optimizes the black-box function'''
        for _ in range(0, iterations):
            X_next = self.acquisition()[0]
            if X_next in self.gp.X:
                break
            self.gp.update(
                X_next,
                self.f(X_next)
            )
        a = np.argmin(
            self.gp.Y
        ) if self.minimize is True else np.argmax(self.gp.Y)
        X_opt = self.gp.X[a]
        Y_opt = self.gp.Y[a]
        return X_opt, Y_opt
