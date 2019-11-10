import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


data = pd.read_csv('C:\\Users\\Connor\\Dropbox\\School\\11 - Applying to First Job\\Sample Take Home\\Beer Reviews Data\\beer_reviews_cleaned.csv')
data['review_overall'].describe()

num_col = ['review_overall',
    'review_aroma',
    'review_appearance',
    'review_palate',
    'review_taste',
    'beer_abv']
# We'll try to predict the review_overall
num_col.remove('review_overall')

X = data[num_col].to_numpy()
Y = data['review_overall'].to_numpy()

### Normalize the data
Y = Y.reshape(-1,1)
from sklearn.preprocessing import StandardScaler
scaler_X = StandardScaler()
scaler_X.fit(X)
scaler_Y = StandardScaler()
scaler_Y.fit(Y)
# scaler.mean_
# scaler.scale_ # Standard Deviation
# pickle.dump({'scaler_X', 'scaler_Y'}, open('Q03b - scalers.p', 'wb'))

X = scaler_X.transform(X)
Y = scaler_Y.transform(Y)

### Linear Regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

lin_reg = LinearRegression()
lin_reg.fit(X,Y)
# import pickle
# pickle.dump(lin_reg, open('Q03a - Linear Regression.p', 'wb'))

predictions = lin_reg.predict(X)
lin_rmse = np.sqrt(mean_squared_error(Y, predictions))
lin_rmse
# rmse = 0.411

from sklearn.metrics import r2_score
r2_score(Y, predictions)
# r^2 score = 0.670
# About 67% of the variation in the review_overall is given by variation in the feature_importances
# In summary, this might tell us which features can increase/decrease review_overall,
# but it probably can't tell us by how much.

### Create plot of coefficients to visualize results
clr_list = []
# for ind in range(len(lin_reg.coef_)):
#     if lin_reg.coef_[ind]>0:
#         clr_list.append('Green')
#     else:
#         clr_list.append('Red')
for ind in lin_reg.coef_[0]>0:
    if ind:
        clr_list.append('Green')
    else:
        clr_list.append('Red')
clr_list

fig = plt.figure(figsize=(17,4))
plt.bar(np.arange(len(lin_reg.coef_[0])), lin_reg.coef_[0], color=clr_list, tick_label=num_col)
plt.title('Linear Regression Coefficients')
# plt.savefig('Q03a - Figure 1 - Linear Regression Coefficients.png')


### Check distributio of residuals
Y_pred = lin_reg.predict(X)
plt.hist(Y_pred - Y, bins=20)
plt.title('Linear Regression Residuals')
plt.xlabel('Residual in Standard Deviations from Mean')
# plt.savefig('Q03a - Figure 2 - Linear Regression Residuals.png')

np.sum( np.logical_and(np.greater_equal(Y_pred-Y, -1), np.less_equal(Y_pred-Y, 1)) ) / Y.size
# 92.2% of predictions are within plus or minus 1 standard deviation
np.sum( np.logical_and(np.greater_equal(Y_pred-Y, -0.5), np.less_equal(Y_pred-Y, 0.5)) ) / Y.size
# 66.1% of predictions are within plus or minus a half a standard deviation
