#!/usr/bin/env python3
'''11. GMM'''
import sklearn.mixture


def gmm(X, k):
    '''calculates a GMM from a dataset'''
    GMM = sklearn.mixture.GaussianMixture(n_components=k)
    tmp = GMM.fit(X)
    pi = tmp.weights_
    m = tmp.means_
    S = tmp.covariances_
    clss = GMM.predict(X)
    bic = GMM.bic(X)
    return pi, m, S, clss, bic
