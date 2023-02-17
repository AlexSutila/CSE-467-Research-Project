#!/bin/python3

import pandas as pd
import numpy as np

np.random.seed(8)
df = pd.read_csv('data/USCensus1990.csv')

shuffle = np.random.permutation(df.index)
half = int(len(shuffle) / 2)

idx_1, idx_2 = shuffle[:half], shuffle[half:]
df1, df2 = df.loc[idx_1], df.loc[idx_2]

df1.to_csv('data/USCensus1990_1.csv', index=False)
df2.to_csv('data/USCensus1990_2.csv', index=False)

