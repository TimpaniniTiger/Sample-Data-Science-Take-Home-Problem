import pickle
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

### Load linear regression
lin_reg = pickle.load(open('Q03a - Linear Regression.p', 'rb'))
### Get coefficients
orig_coef = [x for x in lin_reg.coef_[0]]
mod_coef = [x for x in lin_reg.coef_[0]]
# Aroma is the first coefficient, appearance is the second

### Assume aroma and appearance are twice as important
mult = 2
mod_coef[0] = mult*mod_coef[0]
mod_coef[1] = mult*mod_coef[1]

### Aggregate data into beers
data = pd.read_csv('C:\\Users\\Connor\\Dropbox\\School\\11 - Applying to First Job\\Sample Take Home\\Beer Reviews Data\\beer_reviews_cleaned.csv')
data.head()

### Create same table as in Q01, but using review_overall this time.
beer_data = data.groupby('beer_name', as_index=False).agg(np.mean)
count_data = data.groupby('beer_name').agg(lambda x: x.size).rename(columns={'<lambda_0>':'count'})
# count_data = data['review_time'].groupby('beer_name').agg(lambda x: x.size).rename(columns={'<lambda_0>':'count'})
# # beer_data = data.groupby('beer_name', as_index=False).agg([np.mean, lambda x: x.size]).rename(columns={'<lambda_0>':'count'})
# beer_data.head()
# count_data
# count_data.rename(index=1, columns={'brewery_name:count'})
# count_data['brewery_name'].rename(columns={'brewery_name:count'}).head()
# count_data['brewery_name'].to_numpy()
beer_data['Count'] = pd.Series(count_data['brewery_name'].to_numpy(), index=beer_data.index)
beer_data[['beer_name', 'Count']].head()

num_col = ['review_overall',
    'review_aroma',
    'review_appearance',
    'review_palate',
    'review_taste',
    'beer_abv']
num_col.remove('review_overall')

X = beer_data[num_col].to_numpy()

### Use linear regression to predict new (normalized) ratings
Y_orig = np.matmul(X, orig_coef)
Y_mod = np.matmul(X, mod_coef)

beer_data['Original Predicted Rating'] = pd.Series(Y_orig, index=beer_data.index)
beer_data['Modified Predicted Rating'] = pd.Series(Y_mod, index=beer_data.index)


beer_data[['beer_name', 'Original Predicted Rating', 'Modified Predicted Rating', 'Count']].sort_values(by='Modified Predicted Rating', ascending=False).head(n=10)
beer_data[['beer_name', 'Original Predicted Rating', 'Modified Predicted Rating', 'Count']].sort_values(by='Modified Predicted Rating', ascending=False).head(n=10).to_csv(path_or_buf='Q04 - Modified Predicted Rating.csv')

beer_data[['beer_name', 'Original Predicted Rating', 'Modified Predicted Rating', 'Count']].sort_values(by='Original Predicted Rating', ascending=False).head(n=10)
beer_data[['beer_name', 'Original Predicted Rating', 'Modified Predicted Rating', 'Count']].sort_values(by='Original Predicted Rating', ascending=False).head(n=10).to_csv(path_or_buf='Q04 - Original Predicted Rating.csv')

### Plot Linear Coefficients
clr_list = []
# for ind in range(len(lin_reg.coef_)):
#     if lin_reg.coef_[ind]>0:
#         clr_list.append('Green')
#     else:
#         clr_list.append('Red')
# for ind in lin_reg.coef_[0]>0:
#     if ind:
#         clr_list.append('Green')
#     else:
#         clr_list.append('Red')
# clr_list
clr_list = ['#008080', '#008080', 'Green', 'Green', 'Red']

fig = plt.figure(figsize=(17,4))
plt.bar(np.arange(len(mod_coef)), mod_coef, color=clr_list, tick_label=num_col)
plt.title('Modified Linear Coefficients')
plt.xlabel('review_aroma and review_appearance have been doubled.')
# plt.savefig('Q04 - Figure 2 - Modified Linear Coefficients.png')
