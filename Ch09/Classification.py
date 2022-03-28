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
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [10, 10]

# %%
import cesium
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

from cesium import datasets
from cesium import featurize as ft

import scipy
from scipy.stats import pearsonr, spearmanr
from scipy.stats import skew

import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# %%
print(cesium.__version__)
print(xgb.__version__)
print(scipy.__version__)
print(sklearn.__version__)

# %% [markdown]
# ## Load data and generate some features of interest

# %%
eeg = datasets.fetch_andrzejak()

# %%
type(eeg)

# %%
eeg.keys()

# %% [markdown]
# ### Visually inspect

# %%
plt.subplot(3, 1, 1)
plt.plot(eeg["measurements"][0])
plt.legend(eeg['classes'][0])
plt.subplot(3, 1, 2)
plt.plot(eeg["measurements"][300])
plt.legend(eeg['classes'][300])
plt.subplot(3, 1, 3)
plt.plot(eeg["measurements"][450])
plt.legend(eeg['classes'][450])

# %%
type(eeg["measurements"][0])

# %%
type(eeg)

# %%
eeg.keys()

# %%
type(eeg['measurements'])

# %%
len(eeg['measurements'])

# %%
eeg['measurements'][0].shape

# %% [markdown]
# ## Generate the features

# %%
# from cesium import featurize as ft
# features_to_use = ["amplitude",
#                    "percent_beyond_1_std",
#                    "percent_close_to_median",
#                   "skew",
#                   "max_slope"]
# fset_cesium = ft.featurize_time_series(times=eeg["times"],
#                                               values=eeg["measurements"],
#                                               errors=None,
#                                               features_to_use=features_to_use,
#                                              scheduler = None)

# %%
fset_cesium = pd.read_csv("data/full_eeg_data_features.csv", header = [0, 1])

# %%
fset_cesium.head()

# %%
# fset_cesium.to_csv("full_eeg_data_features.csv")

# %%
fset_cesium.shape

# %% [markdown]
# ## Exercise: validate/calculate these features by hand
# #### look up feature definitions here: http://cesium-ml.org/docs/feature_table.html
# confirm the values by hand coding these features for the first EEG measurement
# (that is eeg["measurements"][0])

# %%
ex = eeg["measurements"][0]

# %%
ex_mean = np.mean(ex)
ex_std  = np.std(ex)

# %%
# amplitude
(np.max(ex) - np.min(ex)) / 2

# %%
 
siz = len(ex)
ll = ex_mean - ex_std
ul = ex_mean + ex_std

quals = [i for i in range(siz) if ex[i] < ll or ex[i] > ul]
len(quals)/len(ex)

# %%
# percent_close_to_median
# Percentage of values within window_frac*(max(x)-min(x)) of median.
# find the source code here:
# https://github.com/cesium-ml/cesium/blob/master/cesium/features/common_functions.py
# window frac = 0.1
window = 0.1 * (np.max(ex) - np.min(ex))
np.where(np.abs(ex_mean - ex) < window)[0].shape[0] / ex.shape[0]

# %%
## skew
print(skew(ex))
plt.hist(ex)

# %%
## max slope
## again check definition : https://github.com/cesium-ml/cesium/blob/master/cesium/features/common_functions.py
times = eeg["times"][0]
np.max(np.abs(np.diff(ex)/np.diff(times)))

# %%
plt.hist(fset_cesium.iloc[:, 1])

# %%
fset_cesium['classes'] = eeg['classes']

# %%
fset_cesium.columns = fset_cesium.columns.droplevel(-1)

# %%
fset_cesium.groupby('classes')['amplitude'].hist()

# %%
fset_cesium['amplitude'].hist(by=fset_cesium['classes'])

# %%
fset_cesium['max_slope'].hist(by=fset_cesium['classes'])

# %% [markdown]
# ## Prepare data for training

# %%
X_train, X_test, y_train, y_test = train_test_split(
     fset_cesium.iloc[:, 1:6].values, eeg["classes"], random_state=21)

# %% [markdown]
# ## Try a random forest with these features

# %%
clf = RandomForestClassifier(n_estimators=10, max_depth=3,
                              random_state=21)

# %%
clf.fit(X_train, y_train)

# %%
clf.score(X_train, y_train)

# %%
clf.score(X_test, y_test)

# %%
np.unique(y_test, return_counts=True)

# %%
y_test

# %%
y_test.shape

# %%
y_train.shape

# %% [markdown]
# ## Try XGBoost with these features

# %%
model = xgb.XGBClassifier(n_estimators=10, max_depth=3,
                              random_state=21)
model.fit(X_train, y_train)

# %%
model.score(X_test, y_test)

# %%
model.score(X_train, y_train)

# %%
xgb.plot_importance(model)

# %% [markdown]
# ## Time Series Forecasting with Decision Trees

# %%
ap = pd.read_csv("data/AirPassengers.csv", parse_dates=[0])

# %%
ap.head()

# %%
ap.set_index('Month', inplace=True)

# %%
ap.head()

# %%
plt.plot(ap)

# %%
plt.plot(np.diff(np.log(ap.values[:, 0])))

# %%
ts = np.diff(np.log(ap.values[:, 0]))

# %% [markdown]
# ## Exercise: now that we have 1 time series, how can we convert it to many samples?

# %%
NSTEPS = 12

# %%
ts.shape

# %%
vals = np.hstack([np.expand_dims(np.array(ts, dtype = np.float32), axis = 1) for _ in range(NSTEPS )])

# %%
ts[0:NSTEPS]

# %%
vals.shape

# %%
nrow = vals.shape[0]
for lag in range(1, vals.shape[1]):
    vals[:(nrow - lag),lag] = vals[lag:,lag]
    vals[(nrow - lag):, lag] = np.nan

# %%
vals

# %%
vals = vals[:(vals.shape[0] - NSTEPS + 1), :]

# %%
vals.shape

# %%
vals[-1]

# %%
ts[-NSTEPS:]

# %%
vals.shape

# %% [markdown]
# ## Exercise: now that we have the time series broken down into a set of samples, how to featurize?

# %%
measures = [vals[i][0:(NSTEPS - 1)] for i in range(vals.shape[0])]

# %%
times = [[j for j in range(NSTEPS - 1)] for i in range(vals.shape[0])]

# %%
measures[0]

# %%
len(measures[0])

# %%
features_to_use = [
                   "amplitude",
                   "percent_beyond_1_std",
                   "skew",
                   "max_slope",
                   "percent_amplitude"]
fset_ap = ft.featurize_time_series(times=times,
                                    values=measures,
                                    errors=None,
                                    features_to_use=features_to_use,
                                    scheduler = None)

# %%
fset_ap.columns = fset_ap.columns.droplevel(-1)

# %%
fset_ap.head()

# %%
plt.hist(fset_ap.amplitude)

# %%
plt.hist(fset_ap.percent_amplitude)

# %%
plt.hist(fset_ap['skew'])

# %% [markdown]
# ## Exercise: can you fit an XGBRegressor to this problem? Let's use the first 100 'time series' as the training data

# %%
outcomes = vals[:, -1]

# %%
X_train, y_train = fset_ap.iloc[:100, :], outcomes[:100]
X_test, y_test   = fset_ap.iloc[100:, :], outcomes[100:]

# %%
X_train.shape

# %%
model = xgb.XGBRegressor(n_estimators=20, max_depth=2,
                              random_state=21)

# %%
eval_set = [(X_test, y_test)]
model.fit(X_train, y_train, eval_metric="rmse", eval_set=eval_set, verbose=True)

# %% [markdown]
# ### RMSE can be hard to digest .... How does the model perform?

# %%
plt.scatter(model.predict(X_test), y_test)

# %%
plt.scatter(model.predict(X_train), y_train)

# %%
pearsonr(model.predict(X_train), y_train)

# %%
pearsonr(model.predict(X_test), y_test)

# %%
xgb.plot_importance(model)

# %% [markdown]
# ### What went wrong? Let's revisit the feature set

# %%
fset_ap.head()

# %%
plt.plot(vals[0])
plt.plot(vals[1])
plt.plot(vals[2])

# %% [markdown]
# ## We need to find a way to generate features that encode positional information

# %% [markdown]
# ### now we will generate our own features

# %%
vals.shape

# %%
feats = np.zeros( (vals.shape[0], 6), dtype = np.float32)
for i in range(vals.shape[0]):
    feats[i, 0] = np.where(vals[i] == np.max(vals[i]))[0][0]
    feats[i, 1] = np.where(vals[i] == np.min(vals[i]))[0][0]
    feats[i, 2] = feats[i, 0] - feats[i, 1]
    feats[i, 3] = np.max(vals[i][-3:])
    feats[i, 4] = vals[i][-1] - vals[i][-2]
    feats[i, 5] = vals[i][-1] - vals[i][-3]

# %%
feats[0:3]

# %% [markdown]
# ### How do these look compared to the first set of features?

# %%
pd.DataFrame(feats[0:3])

# %%
X_train, y_train = feats[:100, :], outcomes[:100]
X_test, y_test   = feats[100:, :], outcomes[100:]

# %%
model = xgb.XGBRegressor(n_estimators=20, max_depth=2,
                              random_state=21)
eval_set = [(X_test, y_test)]
model.fit(X_train, y_train, eval_metric="rmse", eval_set=eval_set, verbose=True)

# %%
plt.scatter(model.predict(X_test), y_test)

# %%
print(pearsonr(model.predict(X_test), y_test))
print(spearmanr(model.predict(X_test), y_test))

# %%
plt.scatter(model.predict(X_train), y_train)

# %%
print(pearsonr(model.predict(X_train), y_train))
print(spearmanr(model.predict(X_train), y_train))

# %%
