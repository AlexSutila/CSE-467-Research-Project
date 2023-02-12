#!/bin/python3

import pandas as pd

df = pd.read_csv('data/USCensus1990.csv')
df1, df2 = df.sample(frac=0.5), df.drop(df.sample(frac=0.5).index)

df1.to_csv('data/USCensus1990_1.csv')
df2.to_csv('data/USCensus1990_2.csv')

