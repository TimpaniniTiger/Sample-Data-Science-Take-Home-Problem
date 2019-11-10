import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp


data = pd.read_csv('C:\\Users\\Connor\\Dropbox\\School\\11 - Applying to First Job\\Sample Take Home\\Beer Reviews Data\\beer_reviews_cleaned.csv')
data.head()

### Create same table as in Q01, but using review_overall this time.
review_data = data[['beer_name','review_overall']].groupby('beer_name').agg([np.mean, np.median, np.std, lambda x: x.size]).rename(columns={'<lambda_0>':'count'})
review_data.head()
# data.sort_values(by='beer_abv', ascending=False).head()
review_data.review_overall.sort_values(by='mean', ascending=False).head(n=30)
review_data.review_overall
review_data.review_overall['std'].hist()


review_data.review_overall[review_data.review_overall['count']>5].sort_values(by='mean', ascending=False).head(n=10)

review_data.review_overall[review_data.review_overall['count']>5].sort_values(by='mean', ascending=False).head(n=10).to_csv(path_or_buf='Q02 - top beers data.csv')


# # This counts all the beers with counts equal to a specified number in the top ##
# sum(review_data.review_overall[['mean','count']].sort_values(by='mean', ascending=False).head(30)['count']==1)
review_data.review_overall['std'].describe()
