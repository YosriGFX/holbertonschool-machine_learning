#!/usr/bin/env python3
'''Moving Average'''


def moving_average(data, beta):
    '''Function that calculates
    the exponential weighted
    moving average of a data set'''
    move_av = []
    vt = 0
    for i in range(len(data)):
        vt = (
            vt * beta + data[i] * (1-beta)
        )
        average = vt / (
            1 - beta ** (i + 1)
        )
        move_av.append(average)
    return move_av
