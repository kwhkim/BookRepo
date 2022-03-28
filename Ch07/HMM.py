# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: rtopython4-pip
#     language: python
#     name: rtopython4-pip
# ---

# %%
# %matplotlib inline

width = 6
height = 3
import matplotlib
matplotlib.rcParams['figure.figsize'] = [width, height]

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import hmmlearn

from hmmlearn.hmm import GaussianHMM

# %%
print(pd.__version__)
print(np.__version__)
print(hmmlearn.__version__)

# %% [markdown]
# ## Look at the data

# %%
nile = pd.read_csv("Nile.csv", index_col = 0)

# %%
nile.head()

# %%
plt.plot(nile.year, nile.val)

# %% [markdown]
# ## Let's take a look at the hmmlearn API

# %%
vals = np.expand_dims(nile.val.values, 1)
n_states = 2
model = GaussianHMM(n_components=n_states, n_iter=100).fit(vals)
hidden_states = model.predict(vals)

# %%
np.bincount(hidden_states)

# %%
plt.plot(hidden_states)


# %% [markdown]
# ## Exercise: how can we package this more conveniently?

# %%

def fitHMM(vals, n_states):
    vals = np.reshape(vals,[len(vals),1])
    
    # fit Gaussian HMM to Q
    model = GaussianHMM(n_components=n_states, n_iter=100).fit(vals)
     
    # classify each observation as state 0 or 1
    hidden_states = model.predict(vals)
 
    # fit HMM parameters
    mus = np.squeeze(model.means_)
    sigmas = np.squeeze(np.sqrt(model.covars_))
    transmat = np.array(model.transmat_)
    print(mus)
    print(sigmas)
    
#     # re-order parameters in ascending order of mean of underlying distribution
#     idx      = np.argsort(mus)
#     mus      = mus[idx]
#     sigmas   = sigmas[idx]
#     transmat = transmat[idx, :][:, idx]
    
#     state_dict = {}
#     states = [i for i in range(n_states)]
#     for i in idx:
#         state_dict[i] = states[idx[i]]
    
#     relabeled_states = [state_dict[h] for h in hidden_states]
    relabeled_states = hidden_states
    return (relabeled_states, mus, sigmas, transmat, model)


# %%
hidden_states, mus, sigmas, transmat, model = fitHMM(nile.val.values, 2)


# %% [markdown]
# ## Exercise: how might we be able to plot this more sensibly?

# %%
def plot_states(ts_vals, states, time_vals):
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Year)')
    ax1.set_ylabel('Nile river flow',         color=color)
    ax1.plot(time_vals, ts_vals,      color=color)
    ax1.tick_params(axis='y',            labelcolor=color)

    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Hidden state', color=color)  
    ax2.plot(time_vals,states,     color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  
    plt.show()


# %%
plot_states(nile.val, hidden_states, nile.year)

# %% [markdown]
# ## Exercise: can we improve on the analysis above?

# %% [markdown]
# ### Cut off the 'special' region

# %%
np.where(hidden_states == 0)

# %%

# %%

# %%
hidden_states, mus, sigmas, transmat, model = fitHMM(nile.val.values, 3)


# %%
plot_states(nile.val, hidden_states, nile.year)

# %%
mus

# %%
np.set_printoptions(precision = 3, suppress = True)

# %%
transmat

# %%
mus

# %% [markdown]
# ## Exercise: generate new synthetic data from the model and then fit it with a fresh HMM model

# %% [markdown]
# #### Easy to sample from an existing HMM model

# %%
res = np.squeeze(model.sample(1000)[0])

# %%
plt.plot(res)

# %% [markdown]
# #### Then refit

# %%
hidden_states, mus, sigmas, transmat, model = fitHMM(res, 3)


# %%
def plot_states_no_time(ts_vals, states):
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Time)')
    ax1.set_ylabel('Value',        color=color)
    ax1.plot(ts_vals,              color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Hidden state', color=color)  
    ax2.plot(states,        color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  
    plt.show()


# %%
plot_states_no_time(res[1:100], hidden_states[1:100])

# %%
transmat

# %%

# %%

# %%
