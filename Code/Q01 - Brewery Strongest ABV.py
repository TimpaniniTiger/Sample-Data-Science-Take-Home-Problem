import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp


data = pd.read_csv('C:\\Users\\Connor\\Dropbox\\School\\11 - Applying to First Job\\Sample Take Home\\Beer Reviews Data\\beer_reviews_cleaned.csv')

data.head()


### Simple examples from pandas documentation to show the concept.
# df = pd.DataFrame([[1, 2, 3],[4, 5, 6],[7, 8, 9],[np.nan, np.nan, np.nan]],columns=['A', 'B', 'C'])
# df.head()
# df2 = df.agg(["mean", "count"], axis="columns")
# df2
#
#
# df = pd.DataFrame({'Animal': ['Falcon', 'Falcon','Parrot', 'Parrot'],'Max Speed': [380., 370., 24., 26.]})
# df.head()
# df.groupby('Animal').head()
# df.groupby('Animal').mean()
# df.groupby('Animal').count()

# data[['brewery_name','beer_abv']].groupby('brewery_name').mean().rename(columns={"beer_abv":'beer_abv_mean'}).head()
# data[['brewery_name','beer_abv']].groupby('brewery_name').count().rename(columns={'beer_abv':'beer_abv_count'}).head()
# Apparently you can also do this all at once with one function.

abv_data = data[['brewery_name','beer_abv']].groupby('brewery_name').agg([np.mean, np.median, np.std, lambda x: x.size]).rename(columns={'<lambda_0>':'count'})
abv_data.head()
# data.sort_values(by='beer_abv', ascending=False).head()
abv_data.beer_abv.sort_values(by='mean', ascending=False).head(n=10)
abv_data.beer_abv.sort_values(by='median', ascending=False).head(n=10)
abv_data.beer_abv.sort_values(by='mean', ascending=False).head(n=10).to_csv(path_or_buf='abv data.csv')
