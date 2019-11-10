import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import xgboost as xgb
import pickle

data = pd.read_csv('C:\\Users\\Connor\\Dropbox\\School\\11 - Applying to First Job\\Sample Take Home\\Beer Reviews Data\\beer_reviews_cleaned.csv')

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
Y = Y.reshape(-1,1)

### Create a set of training data and a set of testing data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2, random_state=1)

### Normalize the data
from sklearn.preprocessing import StandardScaler
scaler_X = StandardScaler()
scaler_X.fit(X_train)
scaler_Y = StandardScaler()
scaler_Y.fit(Y_train)
# scaler.mean_
# scaler.scale_ # Standard Deviation
# pickle.dump({scaler_X, scaler_Y}, open('Q03b - scalers 2.p', 'wb'))

X_train = scaler_X.transform(X_train)
X_test = scaler_X.transform(X_test)
Y_train = scaler_Y.transform(Y_train)
Y_test = scaler_Y.transform(Y_test)

X_train.shape
Y_train.shape

# D_train = xgb.DMatrix(X_train, label=Y_train)
# D_test = xgb.DMatrix(X_test, label=Y_test)

# ### Basic test to make sure everything is working
# Very small
# xgb_model = xgb.XGBRegressor(learning_rate=0.1, max_depth=3, objective='reg:squarederror', tree_method='gpu_hist')
# Note: using gpu DRASTICALLY reduces training time, maybe by a factor of 10
xgb_model = xgb.XGBRegressor(learning_rate=0.1, max_depth=4, n_estimators=50, base_score=0, objective='reg:squarederror', tree_method='gpu_hist')


# More, but still inaccurate
# xgb_model = xgb.XGBRegressor(learning_rate=0.1, max_depth=5, n_estimators=1000, objective='reg:squarederror', tree_method='gpu_hist')
# xgb_model = xgb.XGBRegressor(learning_rate=0.1, max_depth=10, objective='reg:squarederror', tree_method='gpu_hist')
# xgb_model = xgb.XGBRegressor(learning_rate=0.1, max_depth=6, reg_lambda=0, objective='reg:squarederror', tree_method='gpu_hist')
# xgb_model = xgb.XGBRegressor(learning_rate=0.1, max_depth=5, base_score=0, n_estimators=1000, objective='reg:squarederror', tree_method='gpu_hist')


# Way too much
# xgb_model = xgb.XGBRegressor(learning_rate=0.01, max_depth=4, min_child_weight=6, subsample=0.8, colsample_bytree=0.8, n_estimators=5000, objective='reg:squarederror', tree_method='gpu_hist')
# xgb_model = xgb.XGBRegressor(learning_rate=0.01, max_depth=6, base_score=0, reg_lambda=.01, min_child_weight=6, subsample=0.8, colsample_bytree=0.8, n_estimators=5000, objective='reg:squarederror', tree_method='gpu_hist')



xgb_model.fit(X_train, Y_train)
# pickle.dump(xgb_model, open('Q03b - xgb_model 3.p', 'wb'))


### Test accuracy
xgb_model.score(X_train, Y_train)
xgb_model.score(X_test, Y_test)

Y_test_pred = xgb_model.predict(X_test)
from sklearn.metrics import mean_squared_error as mse
mse(Y_test, Y_test_pred)
Y_test.shape
Y_test_pred.shape
Y_test_pred.reshape(-1,1).shape
Y_test_pred = Y_test_pred.reshape(-1,1)
np.sum(np.power((Y_test - Y_test_pred), 2))
np.mean(np.power((Y_test - Y_test_pred), 2))
np.mean(np.abs(Y_test - Y_test_pred))
# Each prediction is roughly half a standard deviation off on average.

plt.hist(Y_test_pred - Y_test, bins=20, color='#b4531f')
plt.title('XGBoost Residuals')
plt.xlabel('Residual in Standard Deviations from Mean')
# plt.savefig('Q03b - Figure 1 - XGBoost Residuals.png')

np.sum(np.greater_equal(Y_test_pred-Y_test, -1))
np.sum(np.less_equal(Y_test_pred-Y_test, 1))
np.sum( np.logical_and(np.greater_equal(Y_test_pred-Y_test, -1), np.less_equal(Y_test_pred-Y_test, 1)) ) / Y_test.size
# 92.6% of predictions are within plus or minus 1 standard deviation
np.sum( np.logical_and(np.greater_equal(Y_test_pred-Y_test, -0.5), np.less_equal(Y_test_pred-Y_test, 0.5)) ) / Y_test.size
# 67.7% of predictions are within plus or minus a half a standard deviation

### Compare to Linear Regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train,Y_train)
Y_test_pred = lin_reg.predict(X_test)
plt.hist(Y_test_pred - Y_test, bins=20)
np.sum( np.logical_and(np.greater_equal(Y_test_pred-Y_test, -1), np.less_equal(Y_test_pred-Y_test, 1)) ) / Y_test.size
# 92.19%
np.sum( np.logical_and(np.greater_equal(Y_test_pred-Y_test, -0.5), np.less_equal(Y_test_pred-Y_test, 0.5)) ) / Y_test.size
# 66.02%
# Both basically the same; this simple XGBoost tree is basically the same as a linear regression.
