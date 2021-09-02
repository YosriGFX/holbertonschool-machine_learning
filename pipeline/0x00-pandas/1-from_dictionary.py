#!/usr/bin/env python3
'''1. From Dictionary'''
import pandas as pd
import numpy as np


df = pd.DataFrame(
    {
        "First": np.array(
            [0.0, 0.5, 1.0, 1.5],
            dtype="float32"
        ),
        "Second": np.array(
            ["one", "two", "three", "four"]
        )
    },
    index=['A', 'B', 'C', 'D']
)
