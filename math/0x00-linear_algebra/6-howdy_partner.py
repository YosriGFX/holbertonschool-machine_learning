#!/usr/bin/env python3
'''Howdy Partner'''


def cat_arrays(arr1, arr2):
    '''cat arrays'''
    return [
        arr1[a] for a in range(len(arr1))
    ] + [
        arr2[a] for a in range(len(arr2))
    ]
