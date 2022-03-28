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
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [10, 5]

import pdb
import numpy as np

# %%
## inspired by https://rajeshrinet.github.io/blog/2014/ising-model/

### CONFIGURATION
N           = 5 # width of lattice
M           = 5 # height of lattice
temperature = 0.5
BETA        = 1 / temperature


# %%
# initialize system
def initRandState(N, M):
    block = np.random.choice([-1, 1], size = (N, M))
    return block


# %%
def costForCenterState(state, i, j, n, m):
    centerS = state[i, j]
    neighbors = [((i + 1) % n, j), ((i - 1) % n, j),
                 (i, (j + 1) % m), (i, (j - 1) % m)]
    ## notice the % n because we impose periodic boundary conditions
    ## ignore this if it doesn't make sense - it's merely a physical constraint on the system

    interactionE = [state[x, y] * centerS for (x, y) in neighbors]
    return np.sum(interactionE)


def magnetizationForState(state):
    return np.sum(state)


# %%
# mcmc steps
def mcmcAdjust(state):
    n = state.shape[0]
    m = state.shape[1]
    x, y = np.random.randint(0, n), np.random.randint(0, m)
    centerS = state[x, y]
    cost = costForCenterState(state, x, y, n, m)
    if cost < 0:
        centerS *= -1
    elif np.random.random() < np.exp(-cost * BETA):
        centerS *= -1
    state[x, y] = centerS
    return state
    
def runState(state, n_steps, snapsteps = None):
    if snapsteps is None:
        snapsteps = np.linspace(0, n_steps, num = round(n_steps / (M * N * 100)), dtype = np.int32)
    saved_states = []
    sp = 0
    magnet_hist = []
    for i in range(n_steps):
        state = mcmcAdjust(state)
        magnet_hist.append(magnetizationForState(state))
        if sp < len(snapsteps) and i == snapsteps[sp]:
            saved_states.append(np.copy(state))
            sp += 1
    return state, saved_states, magnet_hist



# %%
### RUN A SIMULATION
init_state = initRandState(N, M)
plt.imshow(init_state)


# %%
final_state, states, magnet_hist = runState(init_state, 1000)

# %%
plt.imshow(final_state)

# %%
plt.plot(magnet_hist)

# %% [markdown]
# ## Exercise: Modify the simulation code above to visualize magnetization over time for 100 test runs

# %%
results = []
for i in range(100):
    init_state = initRandState(N, M)
    final_state, states, magnet_hist = runState(init_state, 1000)
    results.append(magnet_hist)

# %%
for mh in results:
    plt.plot(mh,'r', alpha=0.2)

# %% [markdown]
# ## Exercise: generate a curve of absolute value of magnetization at step 100 for temperature for temperatures = 0.1, 0.5, 1, 2, 5, 20, 50, 100, 500

# %%
results = []
temps = [0.1, 0.5, 1, 2, 5, 20, 50, 100, 500]

# mcmc steps
def mcmcAdjust(state, beta):
    n = state.shape[0]
    m = state.shape[1]
    x, y = np.random.randint(0, n), np.random.randint(0, m)
    centerS = state[x, y]
    cost = costForCenterState(state, x, y, n, m)
    if cost < 0:
        centerS *= -1
    elif np.random.random() < np.exp(-cost * beta):
        centerS *= -1
    state[x, y] = centerS
    return state
    
def runState(state, beta, n_steps, snapsteps = None):
    if snapsteps is None:
        snapsteps = np.linspace(0, n_steps, num = round(n_steps / (M * N * 100)), dtype = np.int32)
    saved_states = []
    sp = 0
    magnet_hist = []
    for i in range(n_steps):
        state = mcmcAdjust(state, beta)
        magnet_hist.append(magnetizationForState(state))
        if sp < len(snapsteps) and i == snapsteps[sp]:
            saved_states.append(np.copy(state))
            sp += 1
    return state, saved_states, magnet_hist


res = []
for temp in temps:
    temp_res = []
    for _ in range(20):
        init_state = initRandState(N, M)
        final_state, states, magnet_hist = runState(init_state, 1/temp, 100)
        temp_res.append(abs(magnet_hist[-1]))
    res.append(np.mean(temp_res))

# %%
temps

# %%
plt.plot(temps, res)

# %% [markdown]
# ## Exercise: what might an agent-based model look like when applied to this problem?

# %%
