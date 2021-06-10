#!/usr/bin/env python3
'''12. Agglomerative'''
import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    '''performs agglomerative clustering on a dataset'''
    a = scipy.cluster.hierarchy
    b = a.linkage(X, 'ward')
    clss = a.fcluster(b, t=dist, criterion="distance")
    fig = plt.figure()
    tmp = a.dendrogram(b, color_threshold=dist)
    plt.show()
    return clss
