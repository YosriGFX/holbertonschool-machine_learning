#!/usr/bin/env python3
'''2. From File'''
import pandas as pd
import numpy as np


def from_file(filename, delimiter):
    '''A function that loads data from
    a file as a pd.DataFrame'''
    return pd.read_csv(
        filename,
        delimiter=delimiter
    )
