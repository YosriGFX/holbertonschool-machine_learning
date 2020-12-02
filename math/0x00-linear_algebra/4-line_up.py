#!/usr/bin/env python3
'''Line  Up'''


def add_arrays(arr1, arr2):
    '''add arrays'''
    if len(arr1) != len(arr2):
        return None
    else:
        return [
            arr1[a] + arr2[a]
            for a in range(len(arr1))
        ]
