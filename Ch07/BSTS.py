# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# %matplotlib inline
import matplotlib
matplotlib.rcParams['figure.figsize'] = [8, 3]
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels

import scipy
from scipy.stats import pearsonr

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# %%
print(matplotlib.__version__)
print(pd.__version__)
print(np.__version__)
print(statsmodels.__version__)
print(scipy.__version__)


# %% [markdown]
# ## Obtain and visualize data

# %%
## data obtained from https://datahub.io/core/global-temp#data
df = pd.read_csv("global_temps.csv")
df.head()

# %%
df.Mean[:100].plot()

# %% [markdown]
# ## Exercise: what is wrong with the data and plot above? How can we fix this?

 # %%
 df = df.pivot(index='Date', columns='Source', values='Mean')

# %%
df.head()

# %%
df.GCAG.plot()

# %%
type(df.index)

# %% [markdown]
# ## Exercise: how can we make the index more time aware?

# %%
df.index = pd.to_datetime(df.index)

# %%
type(df.index)

# %%
df.GCAG.plot()

# %%
df['1880']

# %%
plt.plot(df['1880':'1950'][['GCAG', 'GISTEMP']])

# %%
plt.plot(df['1950':][['GISTEMP']])

# %% [markdown]
# ## Exercise: How strongly do these measurements correlate contemporaneously? What about with a time lag?

# %%
plt.scatter(df['1880':'1900'][['GCAG']], df['1880':'1900'][['GISTEMP']])

# %%
plt.scatter(df['1880':'1899'][['GCAG']], df['1881':'1900'][['GISTEMP']])

# %%
pearsonr(df['1880':'1899'].GCAG, df['1881':'1900'].GISTEMP)

# %%
df['1880':'1899'][['GCAG']].head()

# %%
df['1881':'1900'][['GISTEMP']].head()

# %%
min(df.index)

# %%
max(df.index)

# %% [markdown]
# ## Unobserved component model

# %%
train = df['1960':]

# %% [markdown]
# ### model parameters

# %%
# smooth trend model without seasonal or cyclical components
model = {
    'level': 'smooth trend', 'cycle': False, 'seasonal': None, 
}


# %% [markdown]
# ### fitting a model

# %%
# https://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.structural.UnobservedComponents.html
gcag_mod = sm.tsa.UnobservedComponents(train['GCAG'], **model)
gcag_res = gcag_mod.fit()

# %%
fig = gcag_res.plot_components(legend_loc='lower right', figsize=(15, 9));

# %% [markdown]
# ## Plotting predictions

# %%
# Perform rolling prediction and multistep forecast
num_steps = 20
predict_res = gcag_res.get_prediction(dynamic=train['GCAG'].shape[0] - num_steps)

predict = predict_res.predicted_mean
ci = predict_res.conf_int()

# %%
plt.plot(predict)

# %%
plt.scatter(train['GCAG'], predict)

# %%
fig, ax = plt.subplots()
# Plot the results
ax.plot(train['GCAG'], 'k.', label='Observations');
ax.plot(train.index[:-num_steps], predict[:-num_steps], label='One-step-ahead Prediction');

ax.plot(train.index[-num_steps:], predict[-num_steps:], 'r', label='Multistep Prediction');
ax.plot(train.index[-num_steps:], ci.iloc[-num_steps:], 'k--');

# Cleanup the image
legend = ax.legend(loc='upper left');

# %%
fig, ax = plt.subplots()
# Plot the results
ax.plot(train.index[-40:], train['GCAG'][-40:], 'k.', label='Observations');
ax.plot(train.index[-40:-num_steps], predict[-40:-num_steps], label='One-step-ahead Prediction');

ax.plot(train.index[-num_steps:], predict[-num_steps:], 'r', label='Multistep Prediction');
ax.plot(train.index[-num_steps:], ci.iloc[-num_steps:], 'k--');

# Cleanup the image
legend = ax.legend(loc='upper left');

# %% [markdown]
# ## Exercise: consider adding a seasonal term for 12 periods for the model fit above. Does this improve the fit of the model?

# %%
seasonal_model = {
    'level': 'local linear trend',
    'seasonal': 12
}
mod = sm.tsa.UnobservedComponents(train['GCAG'], **seasonal_model)
res = mod.fit(method='powell', disp=False)

# %%
fig = res.plot_components(legend_loc='lower right', figsize=(15, 9));

# %% [markdown]
# ## How does this compare to the original model?

# %%
pearsonr(gcag_res.predict(), train['GCAG'])

# %%
np.mean(np.abs(gcag_res.predict() - train['GCAG']))

# %%
np.mean(np.abs(res.predict() - train['GCAG']))

# %% [markdown]
# ## Explore the seasonality more

# %%
seasonal_model = {
    'level': 'local level',
    'seasonal': 12
}
llmod = sm.tsa.UnobservedComponents(train['GCAG'], **seasonal_model)
ll_level_res = llmod.fit(method='powell', disp=False)

# %%
fig = ll_level_res.plot_components(legend_loc='lower right', figsize=(15, 9));

# %%
np.mean(np.abs(ll_level_res.predict() - train['GCAG']))

# %%
train[:48].GCAG.plot()

# %%

# %% [markdown]
# ## Exercise: a common null model for time series is to predict the value at time t-1 for the value at time t. How does such a model compare to the models we fit here?

# %% [markdown]
# ### Consider correlation

# %%
pearsonr(ll_level_res.predict(), train['GCAG'])

# %%
pearsonr(train['GCAG'].iloc[:-1, ], train['GCAG'].iloc[1:, ])

# %% [markdown]
# ### What about mean absolute error?

# %%
np.mean(np.abs(ll_level_res.predict() - train['GCAG']))

# %%
np.mean(np.abs(train['GCAG'].iloc[:-1, ].values, train['GCAG'].iloc[1:, ].values))

# %%
