#!/usr/bin/env python3
'''9. Fill'''
import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

df.pop("Weighted_Price")
df[['Close']] = df[['Close']].ffill()
df['High'] = df['High'].fillna(df['Close'])
df['Open'] = df['Open'].fillna(df['Close'])
df['Low'] = df['Low'].fillna(df['Close'])
df.fillna({"Volume_(BTC)": 0, "Volume_(Currency)": 0}, inplace=True)

print(df.head())
print(df.tail())
