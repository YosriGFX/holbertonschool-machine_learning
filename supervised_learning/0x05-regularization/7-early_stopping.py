#!/usr/bin/env python3
'''Early Stopping'''


def early_stopping(cost, opt_cost, threshold, patience, count):
    '''Function that determines if you
    should stop gradient descent early'''
    if opt_cost - cost < threshold:
        count += 1
    else:
        count = 0
    return (
        count >= patience, count
    )
