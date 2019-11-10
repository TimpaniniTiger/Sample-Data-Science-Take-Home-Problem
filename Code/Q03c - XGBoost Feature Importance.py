import pickle
import xgboost
import shap
import matplotlib.pyplot as plt
import numpy as np

### Load XGBoost model
model = pickle.load(open('Q03b - xgb_model.p', 'rb'))
model
model.feature_importances_

# ### explain the model's predictions using SHAP values ###
# explainer = shap.TreeExplainer(model)
# shap_values = explainer.shap_values(X)
# np.save('shap_values.npy', shap_values)
#
#
# ### visualuze ###
# shap.force_plot(explainer.expected_value, shap_values[0,:], num_col, matplotlib=True)

### Use feature importance from scikitlearn because SHAP takes too long
ft_imp = model.feature_importances_

num_col = ['review_overall',
    'review_aroma',
    'review_appearance',
    'review_palate',
    'review_taste',
    'beer_abv']
# We'll try to predict the review_overall
num_col.remove('review_overall')

# clr_list=[]
# for ind in ft_imp>0:
#     if ind:
#         clr_list.append('Green')
#     else:
#         clr_list.append('Red')
# clr_list

clr_list = ['#820179', '#820179', '#820179', '#820179', '#820179']
fig = plt.figure(figsize=(17,4))
plt.bar(np.arange(len(ft_imp)), ft_imp, color=clr_list, tick_label=num_col)
plt.title('XGBoost Feature Importance')
plt.xlabel('Note that decision tree feature importance is always positive')
# plt.savefig('Q03c - Figure 1 - Feature Importances.png')
