import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp


data = pd.read_csv('C:\\Users\\Connor\\Dropbox\\School\\11 - Applying to First Job\\Sample Take Home\\Beer Reviews Data\\beer_reviews.csv')
data.head()
# data.head(5).to_csv(path_or_buf='00 - Data Sample.csv')
data.describe()
data
### Check for blanks/NaN
data.count()
# 1-1518829/1586614
# 1586599
# 1-1586599/1586614
# 1-1586266/1586614

# brewery_id, sanity check
data[data['brewery_id']==''].head()
data[pd.isna(data['brewery_id'])].head()
# No blanks or NaN

# brewery_name
temp_string = 'brewery_name'
data[data[temp_string]==''].head()
data[pd.isna(data[temp_string])].head()
data[pd.isna(data['brewery_name'])].brewery_id.unique()
data[data['brewery_id']==27].head()
data[data['brewery_id']==1193].brewery_name.unique()
# It looks like these breweries are mislabled. Change brewer_name to brewery_id.
# temp = data
# temp[temp['brewery_id']==1193]['brewery_name'] = temp[temp['brewery_id']==1193]['brewery_id']


# review_profilename
temp_string = 'review_profilename'
data[data[temp_string]==''].head()
data[pd.isna(data[temp_string])].head()
# Relatively few (less than a tenth of a percent), can probably drop without any issue

# beer_abv
temp_string = 'beer_abv'
data[data[temp_string]==''].head()
data[pd.isna(data[temp_string])].head()
# Represents about 4.2% of all data. Might be possible to reconstruct it.

data[data['beer_beerid']==42964].head()
# Testing various beer_id's shows that if beer_abv is NaN for one entry, it's
# NaN for all of them. Let's see if each beer only appears a few times.

# data[pd.isna(data['beer_abv'])].beer_beerid.count()
data[pd.isna(data['beer_abv'])].count()
data_no_beerabv = data[pd.isna(data['beer_abv'])]
data_no_beerabv.beer_beerid.unique().size
# 67785/17034 # About 4 entries per beer_beerid
# data_no_beerabv.beer_beerid.unique()[0:10]

# Let's go through some beer_id's to see if they're identified elsewhere
for ind in data_no_beerabv.beer_beerid.unique()[0:30]:
    print(ind)
    print(data[data['beer_beerid']==ind].beer_abv.unique())
# Conclusion: Let's drop the rows with missing values. beer_abv is pretty
# important, and the data is legitimately missing.

### Implementing Data Cleaning

# brewery_name - replace missing name with id number
data.loc[data['brewery_id']==1193,['brewery_name']] = '1193'
data[data['brewery_id']==1193].head()
data.loc[data['brewery_id']==27,['brewery_name']] = '27'
data[data['brewery_id']==27].head()


# review_profilename - Drop whole column
data = data.drop(columns=['review_profilename'])

# beer_abv - Drop rows with blank entries
data = data.dropna(subset=['beer_abv'])

# Final check to see if everything works
data.count()



### Performing Anamoly Detection
data.head()
# Which numeric values do we want to perform anamoly detection on?
num_col = ['review_overall',
    'review_aroma',
    'review_appearance',
    'review_palate',
    'review_taste',
    'beer_abv']
num_data = data[num_col].dropna()
# Covariance matrix
Sx = np.cov(np.transpose(num_data.to_numpy())) # Transpose because numpy expects opposite rows and cols
Sx = np.linalg.inv(Sx)
mean = np.mean(num_data.to_numpy(),0)
# We'll be using vectorization to perform linear algebra calculations faster
# Tile mean vector so we can subtract from data
temp2 = np.tile(mean, [num_data.shape[0],1])
diff = num_data.to_numpy() - temp2
temp3 = np.matmul(diff, Sx)
maha_dist = np.sqrt(np.sum(np.multiply(temp3, diff), 1))
num_data['Mahalanobis_Distance'] = pd.Series(maha_dist, index=num_data.index)


plt.hist(maha_dist, bins=30)
plt.title('Mahalnobis Distance')
# plt.savefig('00 - Figure 1 - Mahalanobis Distance.png')

sum(num_data['Mahalanobis_Distance']>10)

# plt.hist(num_data[num_data['Mahalanobis_Distance']<10], bins=30)
temp = num_data[num_data['Mahalanobis_Distance']<10]
plt.hist(temp['Mahalanobis_Distance'].to_numpy(), bins=10)

# This is a little weird that the z-scores aren't clustered around 0.

### Drop all entries with a Mahalanobis_Distance > 10.
data['Mahalanobis_Distance'] = pd.Series(maha_dist, index=num_data.index)
data = data[data['Mahalanobis_Distance']<10]

### Output csv of cleaned data
data.to_csv(path_or_buf='C:\\Users\\Connor\\Dropbox\\School\\11 - Applying to First Job\\Sample Take Home\\Beer Reviews Data\\beer_reviews_cleaned.csv')
