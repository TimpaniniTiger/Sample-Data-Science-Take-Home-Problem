# Introduction

This is the sample take-home problem proposed by TCB Analytics in
<https://tcbanalytics.com/2016/01/29/how-to-hire-and-test-for-data-skills-a-one-size-fits-all-interview-kit/>.
The assignment consists of four problems of increasing abstraction that
is meant to test an applicant’s technical data science programming
abilities along with their ability to clearly communicate their results.

## The Data

The data is a 175mb collection of 1,586,614 beer reviews, with 13
columns. It can be downloaded from
<https://data.world/socialmediadata/beeradvocate>.

The 13 columns are **brewery\_id, brewery\_name, review\_time,
review\_overall, review\_aroma, review\_appearance, review\_profilename,
beer\_style, review\_palate, review\_taste, beer\_name, beer\_abv,** and
**beer\_beerid.**

The most natural way to view this data set is to try to predict
**review\_overall** using the other five numeric columns, the first four
**review\_aroma, review\_appearance, review\_palate, review\_taste**
coming from user supplied reviews rating various aspects of a beer, and
**beer\_abv** which gives the beer’s percent alcohol by volume**.**

| **brewery\_id** | **brewery\_name**       | **review\_time** | **review\_overall** | **review\_aroma** | **review\_appearance** | **review\_profilename** | **beer\_style**                | **review\_palate** | **review\_taste** | **beer\_name**         | **beer\_abv** | **beer\_beerid** |
| --------------- | ----------------------- | ---------------- | ------------------- | ----------------- | ---------------------- | ----------------------- | ------------------------------ | ------------------ | ----------------- | ---------------------- | ------------- | ---------------- |
| 10325           | Vecchio Birraio         | 1234817823       | 1.5                 | 2                 | 2.5                    | stcules                 | Hefeweizen                     | 1.5                | 1.5               | Sausa Weizen           | 5             | 47986            |
| 10325           | Vecchio Birraio         | 1235915097       | 3                   | 2.5               | 3                      | stcules                 | English Strong Ale             | 3                  | 3                 | Red Moon               | 6.2           | 48213            |
| 10325           | Vecchio Birraio         | 1235916604       | 3                   | 2.5               | 3                      | stcules                 | Foreign / Export Stout         | 3                  | 3                 | Black Horse Black Beer | 6.5           | 48215            |
| 10325           | Vecchio Birraio         | 1234725145       | 3                   | 3                 | 3.5                    | stcules                 | German Pilsener                | 2.5                | 3                 | Sausa Pils             | 5             | 47969            |
| 1075            | Caldera Brewing Company | 1293735206       | 4                   | 4.5               | 4                      | johnmichaelsen          | American Double / Imperial IPA | 4                  | 4.5               | Cauldron DIPA          | 7.7           | 64883            |

## The Problems

The take-home consists of the following four problems:

1.  *Which brewery produces the strongest beers by ABV%?*

2.  *If you had to pick 3 beers to recommend using only this data, which
    would you pick?*

3.  *Which of the factors (aroma, taste, appearance, palette) are most
    important in determining the overall quality of a beer?*

4.  *Lastly, if I typically enjoy a beer due to its aroma and
    appearance, which beer style should I try?*

These four problems are given in order of increasing technical and
theoretical sophistication.

# Data Cleanup

## Missing Data

Three columns are missing data, **brewery\_name, review\_profilename,**
and **beer\_abv**. All three columns will necessitate different
strategies.

For the **brewery\_name** column a few breweries are missing a name, but
they have an id number given in the **brewery\_id** column. My solution
is to replace the **brewery\_name** with the **brewery­\_id**.

For the **review\_profilename** column relatively few (less than 0.1%)
entries are missing. However, this column doesn’t seem useful for any of
the four questions, so my solution is to drop this column.

For the **beer\_abv** column a fair amount of data (about 4.2%) is
missing. My first action was to search for other entries with the same
**beer\_name** to see if we could fill in the missing data, but
unfortunately this did not work. This column is vital to the analysis
and there seems to be no way to recover it, so unfortunately my solution
is to remove all missing entries.

## Anomaly Detection

For this stage I used the multidimensional z-score (or Mahalanobis
Distance) on the numeric columns to see how anomalous each entry is,
then drop the outliers.

The Mahalanobis Distance is straightforward to compute via the formula

Sqrt((x-mu)S^-1(x-mu)), where x is the observation, mu is the mean, and
S is the covariance matrix,

D=\\sqrt{(x-\\mu)S^{-1}(x-\\mu)}

. This can be efficiently calculated using vectorization with numpy, as
in the following code.

*Sx = np.cov(np.transpose(num\_data.to\_numpy())) \# Transpose because
numpy expects opposite rows and cols*

*Sx = np.linalg.inv(Sx)*

*mean = np.mean(num\_data.to\_numpy(),0)*

*\# We'll be using vectorization to perform linear algebra calculations
faster*

*\# Tile mean vector so we can subtract from data*

*temp2 = np.tile(mean, \[num\_data.shape\[0\],1\])*

*diff = num\_data.to\_numpy() - temp2*

*temp3 = np.matmul(diff, Sx)*

*maha\_dist = np.sqrt(np.sum(np.multiply(temp3, diff), 1))*

![](media/image1.png)

This figure gives the distribution of z-scores. One immediate concern is
that 0 isn’t the most frequent score, which likely indicates that this
data is not normally distributed. I chose 10 standard deviations as the
cutoff because anything past it was unambiguously an outlier, overall
223 entries, however it could be set lower according to taste.

## Lack of Entries for some Beers/Breweries

One concern that comes up later is that some types of beer and some
breweries only have one or two entries in the data. While technically
not an issue, can one really consider a beer with only one rating “the
best”? Removing these entries might also improve the later regressions,
perhaps by making them less noisy. We postpone such a large-scale
elimination of data without more context about what is acceptable for
the data set’s intended use.

# Q01 – Strongest Beer by ABV%?

This question mainly serves as a technical exercise and is easily
handled by using Pandas’ *groupby* and *aggregate* functions on the
**beer\_abv­** column as in the following code.

abv\_data =
data\[\['brewery\_name','beer\_abv'\]\].groupby('brewery\_name').agg(\[np.mean,
np.median, np.std, lambda x:
x.size\]).rename(columns={'\<lambda\_0\>':'count'})

![](media/image2.png)

This figure shows the beers ordered by their mean **beer­\_abv**: the
columns mean, median, std, and count are all computed for **beer\_abv**.
Breweries with only one or two beers are a little suspect, so taking
this into account my choices for the top five are highlighted. Monk’s
Porter House wasn’t chosen because its standard deviation is unusually
high.

# Q02 – Top Three Beers

In the absence of any other unambiguous indication of beer quality, we
will defer to the **overall\_review** column as the proxy for beer
quality. We can then handle this question like the previous question,
using Pandas’ *groupby* and *aggregate* functions on the
**overall\_review** column as in the following code.

review\_data =
data\[\['beer\_name','review\_overall'\]\].groupby('beer\_name').agg(\[np.mean,
np.median, np.std, lambda x:
x.size\]).rename(columns={'\<lambda\_0\>':'count'})

# ![](media/image3.png)

This figure shows the beers ordered by their mean **overall\_review**:
the columns mean, median, std, and count are all computed for
**overall\_review**. Unlike in the previous question each type of beer
has plenty of entries and relatively low variance (the average standard
deviation among all beers was 0.54) so choosing the top three is
straightforward.

# Q03 – Which of Aroma, Taste, Appearance, or Palette is most Important?

As in the last question, we will use **overall\_review** as our proxy
for beer quality. We’ll start by using linear regression on the numeric
columns because linear regression is simple and the linear coefficients
can easily be interpreted as telling how important the various features
are. We implement the linear regression in Scikit-Learn, and these are
the resulting linear coefficients.

![](media/image4.png)

From this we can see that **review\_taste­** is the most important
quality, with **review\_palate** being about half as important, and
**review\_aroma** and **review\_appearance** being far less important.
While not required for this problem, we can also see that **beer\_abv**
is negative correlated.

![](media/image5.png)

But how good is linear regression? We get an r-squared score of 0.670,
which theoretically means that 67% of the variance in
**review\_overall** is given by variance in the numeric columns. Another
way to tell would be to look at the residuals; 92.2% of the predictions
are within plus/minus one standard deviation (one standard deviation is
0.72 points of **review\_overall**) and 66.1% of the predictions are
within half a standard deviation.

Another model we can try is XGBoost, which implements a boosted gradient
decision tree method. Decision trees have a branch-like structure, where
at each branching point one proceeds along a direction governed by
simple rules depending on the input features. Gradient boosting is a
method of chaining decision tree models together to produce an overall
better model. XGBoost is the current fad for machine learning, in
particular achieving impressive results in the realm of Kaggle.

![](media/image6.png)

With XGBoost 92.6% of the predictions are within plus/minus one standard
deviation and 67.7% of predictions are within half a standard deviation.
Comparing this to linear regression, this is the slightest of increases.
We’re probably better off sticking with linear regression as it’s an
easier model to work with.

Before moving on, a note about which parameters were chosen and how this
same broad behavior was observed. The diagnosis from these residuals is
underfitting: one would naively expect XGBoost to have far superior
performance on this data set. The most important two parameters for
XGBoost are the decision tree depth (i.e. how many decisions can each
tree make) and how many trees to chain together. Even setting both these
parameters to ridiculous levels failed to improve performance. Somewhat
notably the models achieve the same level of performance on the training
set as on the cross-validation set, showing that there’s not much more
for the model to learn from the training data.

What could be causing the lackluster performance of XGBoost? Recall from
the data cleaning section, specifically the Mahalanobis Distance
subsection, where the mode of the z-scores wasn’t zero but was actually
closer to one. This could be an indication that the data is particularly
noisy: XGBoost would then be having trouble finding a single answer
because no single answer might exist.

![](media/image7.png)

XGBoost, being fundamentally a decision tree model, can give us feature
importance by aggregating the decision rules of each branch-point. One
particular quirk of decision tree feature importance is that each
importance is always positive, so the fact that the importance of
**beer\_abv** is positive doesn’t necessarily contradict the importance
given by linear regression. Comparing XGBoost with linear regression
they broadly agree on the relative ranking of importance, but they
differ in the relative difference.

To answer the question, taste seems to be the most important feature.

# Q04 – What if Aroma and Appearance is More Important?

From the previous question we have both a linear regression model and an
XGBoost model to use. Also recall that both methods had roughly equal
accuracy. Here though the interpretability of linear regression wins out
because we can simulate **review\_aroma** and **review\_appearance** by
artificially increasing their linear coefficients and predicting new
review ratings. Here we will arbitrarily multiply both aroma’s and
appearance’s linear coefficients by two to make them both twice as
important.

![](media/image8.png)

![](media/image9.png)

The difference between these tables is which rating we’re sorting by:
the modified rating where aroma and appearance are more important, and
the original rating predicted by linear regression. The top five beers
when aroma and appearance are more important are highlighted in blue.

We can see that making aroma and appearance more important slightly
shifts the rankings, but typically a beer does not change rankings much.
This can be changed by changing the relative importance of aroma and
appearance.

A cause for concern is that each of these beers only has one review in
the data set. One could remove any beer with only one review, but then
one should also remove those beers from the training data for linear
regression. A resolution to this would require more context on the
intended usage of this data set.
