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

# %% [markdown]
# ## Clustering time series for classification

# %%
# %matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [15, 10]

from math import sqrt

from datetime import datetime
import pandas as pd
import numpy as np
import pdb


from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

from sklearn.metrics.pairwise import pairwise_distances
from sklearn import preprocessing
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.cluster import homogeneity_score, completeness_score
from sklearn.metrics.cluster import contingency_matrix
from sklearn.metrics.cluster import homogeneity_score

from dtaidistance import dtw

from collections import Counter

from scipy.stats import pearsonr

# %% [markdown]
# ## The data

# %%
words = pd.read_csv('https://raw.githubusercontent.com/AileenNielsen/TimeSeriesAnalysisWithPython/master/data/50words_TEST.csv',
                   header = None)

# %%
words.rename(columns = {0:'word'}, inplace = True) 

# %%
words.head()

# %% [markdown]
# ## View output

# %%
words.word[1]

# %%
plt.subplot(3, 2, 1)
plt.plot(words.iloc[1, 1:-1])
plt.title("Sample Projection Word " + str(words.word[1]), fontweight = 'bold', y = 0.8, fontsize = 14)
plt.subplot(3, 2, 2)
plt.hist(words.iloc[1, 1:-1], 10)
plt.title("Histogram of Projection Word " + str(words.word[1]), fontweight = 'bold', y = 0.8, fontsize = 14)
plt.subplot(3, 2, 3)
plt.plot(words.iloc[3, 1:-1])
plt.title("Sample Projection Word " + str(words.word[3]), fontweight = 'bold', y = 0.8, fontsize = 14)
plt.subplot(3, 2, 4)
plt.hist(words.iloc[3, 1:-1], 10)
plt.title("Histogram of Projection Word " + str(words.word[3]), fontweight = 'bold', y = 0.8, fontsize = 14)
plt.subplot(3, 2, 5)
plt.plot(words.iloc[5, 1:-1])
plt.title("Sample Projection Word " + str(words.word[11]), fontweight = 'bold', y = 0.8, fontsize = 14)
plt.subplot(3, 2, 6)
plt.hist(words.iloc[5, 1:-1], 10)
plt.title("Histogram of Projection Word " + str(words.word[11]), fontweight = 'bold', y = 0.8, fontsize = 14)
plt.suptitle("Sample word projections and histograms of the projections", fontsize = 18)

# %%

## We can also consider the 2d histogram of a word
x = np.array([])
y = np.array([])

w = 23
selected_words = words[words.word == w]
selected_words.shape

for idx, row in selected_words.iterrows():
    y = np.hstack([y, row[1:271]])
    x = np.hstack([x, np.array(range(270))])
    
fig, ax = plt.subplots()
hist = ax.hist2d(x, y, bins = 50)
plt.xlabel("Time", fontsize = 18)
plt.ylabel("Value", fontsize = 18)

# %% [markdown]
# ## Generate some features

# %%
words.shape

# %%
words_features = words.iloc[:, 1:271]

# %% [markdown]
# ### Create some features from original time series

# %%
times  = []
values = []
for idx, row in words_features.iterrows():
    values.append(row.values)
    times.append(np.array([i for i in range(row.values.shape[0])]))

# %%
len(values)

# %%
# from cesium import featurize
# features_to_use = ["amplitude",
#                    "percent_beyond_1_std",
#                    "percent_close_to_median",
#                    ]
# featurized_words = featurize.featurize_time_series(times=times,
#                                               values=values,
#                                               errors=None,
#                                               features_to_use=features_to_use,
#                                               scheduler = None)

# %%
featurized_words = pd.read_csv("data/featurized_words.csv", header = [0, 1])
featurized_words.columns = featurized_words.columns.droplevel(-1)

# %%
featurized_words.head()

# %%
featurized_words.shape

# %%

# %%
plt.hist(featurized_words.percent_beyond_1_std)

# %% [markdown]
# ### Create some features from histogram

# %%
# times = []
# values = []
# for idx, row in words_features.iterrows():
#     values.append(np.histogram(row.values, bins=10, range=(-2.5, 5.0))[0] + .0001) ## cesium seems not to handle 0s
#     times.append(np.array([i for i in range(9)]))

# %%
# features_to_use = ["amplitude",
#                    "percent_close_to_median",
#                   "skew"
#                   ]
# featurized_hists = featurize.featurize_time_series(times=times,
#                                               values=values,
#                                               errors=None,
#                                               features_to_use=features_to_use,
#                                               scheduler = None)

# %%
# featurized_hists.to_csv("data/featurized_hists.csv")

# %%
featurized_hists = pd.read_csv("data/featurized_hists.csv", header = [0, 1])
featurized_hists.columns = featurized_hists.columns.droplevel(-1)

# %%
featurized_hists.head()

# %%
features = pd.concat([featurized_words.reset_index(drop=True), featurized_hists], axis=1)

# %%
features.head()

# %%
words.shape

# %%
## we also add some of our own features again, to account more for shape
feats = np.zeros( (words.shape[0], 1), dtype = np.float32)
for i in range(words.shape[0]):
    vals = words.iloc[i, 1:271].values
    feats[i, 0] = np.where(vals == np.max(vals))[0][0]

# %%
feats.shape

# %%
features.shape

# %%
features['peak_location'] = feats

# %%
features.head()

# %%
feature_values = preprocessing.scale(features.iloc[:, [1, 2, 3, 5, 6, 7]])

# %%

clustering = AgglomerativeClustering(n_clusters=50, linkage='ward')
clustering.fit(feature_values)
words['feature_label'] = clustering.labels_

# %%
words['feature_label'] = words.feature_label.astype('category')

# %%
## the number of feature labels 
results = words.groupby('word')['feature_label'].agg({'num_clustering_labels': lambda x: len(set(x)),
                                            'num_word_samples':      lambda x: len(x),
                                            'most_common_label':     lambda x: Counter(x).most_common(1)[0][0]})
results.head()

# %%
## the number of feature labels 
results_feats = words.groupby('feature_label')['word'].agg({'num_words': lambda x: len(set(x)),
                                            'num_feat_samples':      lambda x: len(x),
                                            'most_common_word':     lambda x: Counter(x).most_common(1)[0][0]})
results_feats
## note that word 1 = most common in cluster 38

# %%
homogeneity_score(words.word, words.feature_label)
## see definitions in user manual: https://scikit-learn.org/stable/modules/clustering.html#homogeneity-completeness

# %% [markdown]
# ## Dynamic Time Warping Distance Definition

# %%
ts1 = np.sin(np.linspace(1, 10))
ts2 = np.sin(2 * np.linspace(1, 10))
ts3 = np.zeros((50,)) 
plt.plot(ts1)
plt.plot(ts2)
plt.plot(ts3)

# %% [markdown]
# ## Exercise: calculate the Euclidean distance between respective pairs of time series from the 3 time series above

# %%
np.sqrt(np.sum(np.square(ts1 - ts2)))

# %%
np.sqrt(np.sum(np.square(ts1 - ts3)))

# %%
np.sqrt(np.sum(np.square(ts2 - ts3)))

# %%
np.linspace(1,10).shape

# %% [markdown]
# ## Another time series clustering technique that has been recommended is a correlation measure. How does this fair in the case of our sine curves and straigh line?

# %%
np.random.seed(215202)
ts3_noise = np.random.random(ts3.shape)
ts3 = np.zeros((50,)) 
ts3 = ts3 + ts3_noise

# %%
pearsonr(ts1, ts2)

# %%
pearsonr(ts1, ts3)

# %%
pearsonr(ts2, ts3 + np.random.random(ts3.shape))

# %% [markdown]
# ## Exercise: use what we discussed about dynamic programming to code a DTW function

# %%
X = words.iloc[:, 1:271].values


# %%
def distDTW(ts1, ts2):
    DTW       = np.full((len(ts1) + 1, len(ts2) + 1), 0, dtype = np.float32)
    DTW[:, 0] = np.inf
    DTW[0, :] = np.inf
    DTW[0, 0] = 0

    for i in range(1, len(ts1) + 1):
        for j in range(1, len(ts2) + 1):
            idx1 = i - 1 
            idx2 = j - 1
            
            dist               = (ts1[idx1] - ts2[idx2])**2
            min_preceding_dist = min(DTW[i-1, j],DTW[i, j-1], DTW[i-1, j-1])

            DTW[i, j] = dist + min_preceding_dist

    return sqrt(DTW[len(ts1), len(ts2)])


# %% [markdown]
# ## Exercise: does this fix the problem above noted with the sine curves vs. a straight line?

# %%
distDTW(ts1, ts2)

# %%
distDTW(ts1, ts3)

# %%
distDTW(ts2, ts3)

# %%
distDTW(X[0], X[1])

# %%
dtw.distance(X[0], X[1])
## worth checking out: https://github.com/wannesm/dtaidistance

# %%
# p = pairwise_distances(X, metric = distDTW)

# %%
# with open("pairwise_word_distances.npy", "wb") as f:
#     np.save(f, p)

# %%
p = np.load("data/pairwise_word_distances.npy")

# %% [markdown]
# ## Exercise: Try clustering based on dynamic time warping distances

# %%
## We will use hierarchical clustering as a distance agnostic methodology

# %%
clustering = AgglomerativeClustering(linkage='average', n_clusters=50, affinity = 'precomputed') 
## 'average' linkage is good for non Euclidean distance metrics

# %%
labels = clustering.fit_predict(p)

# %%
len(words.word)

# %%
len(labels)

# %% [markdown]
# ## Exercise: How did the clustering perform?

# %%

print(homogeneity_score(words.word, labels))
print(completeness_score(words.word, labels))

# %%
# quoting: https://scikit-learn.org/stable/modules/clustering.html#homogeneity-completeness
# homogeneity: each cluster contains only members of a single class.
# completeness: all members of a given class are assigned to the same cluster.

# %%
res = contingency_matrix(labels, words.word)

# %%
## note difficulties in assessing this given imbalanced dataset
plt.imshow(res)
