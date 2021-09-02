#!/usr/bin/env python3
'''3. Rename'''

import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

df['Datetime'] = pd.to_datetime(
    df.pop('Timestamp'),
    unit='s'
)
df = df[['Datetime', 'Close']]
print(df.tail())
