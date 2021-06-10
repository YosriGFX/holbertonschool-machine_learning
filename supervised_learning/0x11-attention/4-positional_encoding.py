#!/usr/bin/env python3
'''3. Positional Encoding'''
import numpy as np


def positional_encoding(max_seq_len, dm):
    '''calculates the positional encoding for a transformer'''
    pos = np.arange(
        max_seq_len
    )
    PE = pos[:, np.newaxis] * 1 / np.power(
        10000, (
            2 * (
                np.arange(
                    dm
                )[np.newaxis, :] // 2
            )
        ) / np.float32(dm)
    )
    PE[:, 0::2] = np.sin(PE[:, 0::2])
    PE[:, 1::2] = np.cos(PE[:, 1::2])
    return PE
