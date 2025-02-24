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

width = 6
height = 3
import matplotlib
matplotlib.rcParams['figure.figsize'] = [width, height]

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## utilities
import os

## deep learning module
import mxnet as mx

## data processing
import numpy as np
import pandas as pd

## reporting
import perf
from scipy.stats import pearsonr, spearmanr

# %% [markdown]
# ## Configure parameters

# %%
## some hyperparameters we won't tune via command line inputs
DATA_SEGMENTS    = { 'tr': 0.6, 'va': 0.2, 'tst': 0.2}
THRESHOLD_EPOCHS = 2
COR_THRESHOLD    =  0.005

## temporal slicing
WIN              = 24 ##* 7
H                = 3

## model details 
MODEL            = 'rnn_model'
SZ_FILT          = 8
N_FILT           = 10
RNN_UNITS        = 10
SEASONAL_PERIOD  = 24

## training details
GPU              = 0
BATCH_N          = 1024
LR               = 0.0001
DROP             = 0.2
N_EPOCHS         = 30

## data details
DATA_FILE        = 'electricity.diff.txt'
SAVE_DIR         = "resultsDir"

# %% [markdown]
# ## Exercise: Look at the data

# %%
elec = pd.read_csv('electricity.diff.txt')

# %%
elec.head()

# %%
plt.plot(elec.V1)

# %%
plt.plot(elec.V1[:96])


# %% [markdown]
# ## Handy data structures

# %%
## courtesy of https://www.saltycrane.com/blog/2007/11/python-circular-buffer/
class RingBuffer:
    def __init__(self, size):
        self.data = [0 for i in range(size)]

    def append(self, x):
        self.data.pop(0)
        self.data.append(x)

    def get(self):
        return self.data


# %% [markdown]
# ## Data preparation

# %%
################################
## DATA PREPARATION ##
################################

def prepared_data(data_file, win, h, model_name):
    df = pd.read_csv(data_file, sep=',', header=0)
    x  = df.iloc[:, :].values ## need to drop first column as that's an index not a value
    x = (x - np.mean(x, axis = 0)) / (np.std(x, axis = 0)) ## normalize data
    
    if model_name == 'fc_model':
        ## provide first and second step lookbacks in one flat input
        X = np.hstack([x[1:-h], x[0:-(h+1)]])
        Y = x[(h+1):]
        return (X, Y)
    else:    
        # preallocate X and Y data arrays
        # X shape = num examples * time win * num channels (NTC)
        X = np.zeros((x.shape[0] - win - h, win, x.shape[1]))
        # Y shape = num examples * num channels
        Y = np.zeros((x.shape[0] - win - h, x.shape[1]))
        
        for i in range(win, x.shape[0] - h):
            y_i = x[i + h - 1     , :] ## the target value is h steps ahead
            x_i = x[(i - win) : i , :] ## the input data are the previous win steps
            X[i-win] = x_i
            Y[i-win] = y_i

        return (X, Y)


def prepare_iters(data_file, win, h, model, batch_n):
    X, Y = prepared_data(data_file, win, h, model)

    n_tr = int(Y.shape[0] * DATA_SEGMENTS['tr'])
    n_va = int(Y.shape[0] * DATA_SEGMENTS['va'])

    X_tr, X_valid, X_test = X[                      : n_tr], \
                               X[n_tr             : n_tr + n_va], \
                               X[n_tr + n_va : ]
    Y_tr, Y_valid, Y_test = Y[                      : n_tr], \
                               Y[n_tr             : n_tr + n_va], \
                               Y[n_tr + n_va : ]
    
    iter_tr = mx.io.NDArrayIter(data       = X_tr,
                                   label      = Y_tr,
                                   batch_size = batch_n)
    iter_val = mx.io.NDArrayIter(  data       = X_valid,
                                   label      = Y_valid,
                                   batch_size = batch_n)
    iter_test = mx.io.NDArrayIter( data       = X_test,
                                   label      = Y_test,
                                   batch_size = batch_n)

    return (iter_tr, iter_val, iter_test)



# %% [markdown]
# ## Define models

# %%
################
## MODELS ##
################

def fc_model(iter_train, input_feature_shape, X, Y,
             win, sz_filt, n_filter, drop, seasonal_period):
    output = mx.sym.FullyConnected(data=X, num_hidden=20)
    output = mx.sym.Activation(output, act_type = 'relu')
    output = mx.sym.FullyConnected(data=output, num_hidden=10)
    output = mx.sym.Activation(output, act_type = 'relu')
    output = mx.sym.FullyConnected(data=output, num_hidden=321)
    
    loss_grad = mx.sym.LinearRegressionOutput(data=output, label=Y)
    return (loss_grad,
            [v.name for v in iter_train.provide_data],
            [v.name for v in iter_train.provide_label])    
    
def cnn_model(iter_train, input_feature_shape, X, Y,
              win, sz_filt, n_filter, drop, seasonal_period):
    conv_input = mx.sym.reshape(data=X, shape=(0, 1, win, -1)) 
    ## Convolution expects 4d input (N x channel x height x width)
    ## in our case channel = 1 (similar to a black and white image
    ## height = time and width = channels slash electric locations
    
    cnn_output = mx.sym.Convolution(data=conv_input,
                                    kernel=(sz_filt,
                                            input_feature_shape[2]),
                                    num_filter=n_filter)
    cnn_output = mx.sym.Activation(data=cnn_output, act_type='relu')
    cnn_output = mx.sym.reshape(mx.sym.transpose(data=cnn_output,
                                                 axes=(0, 2, 1, 3)),
                                shape=(0, 0, 0)) 
    cnn_output = mx.sym.Dropout(cnn_output, p=drop)
        
    output = mx.sym.FullyConnected(data=cnn_output,
                                   num_hidden=input_feature_shape[2])
    loss_grad = mx.sym.LinearRegressionOutput(data=output, label=Y)
    return (loss_grad,
            [v.name for v in iter_train.provide_data],
            [v.name for v in iter_train.provide_label])    

    
def rnn_model(iter_train, input_feature_shape, X, Y,
              win, sz_filt, n_filter, drop, seasonal_period):
    rnn_cells = mx.rnn.SequentialRNNCell()
    rnn_cells.add(mx.rnn.GRUCell(num_hidden=RNN_UNITS))
    rnn_cells.add(mx.rnn.DropoutCell(drop))
    outputs, _ = rnn_cells.unroll(length=win, inputs=X, merge_outputs=False)
    rnn_output = outputs[-1] # only take value from final unrolled cell for use later
    
    output = mx.sym.FullyConnected(data=rnn_output, num_hidden=input_feature_shape[2])
    loss_grad = mx.sym.LinearRegressionOutput(data=output, label=Y)
    return (loss_grad,
            [v.name for v in iter_train.provide_data],
            [v.name for v in iter_train.provide_label])    

## simplifications to
## https://github.com/apache/incubator-mxnet/blob/master/example/multivariate_time_series/src/lstnet.py
def simple_lstnet_model(iter_train,  input_feature_shape, X, Y,
                        win, sz_filt, n_filter, drop, seasonal_period):
    ## must be 4d or 5d to use padding functionality
    conv_input = mx.sym.reshape(data=X, shape=(0, 1, win, -1)) 

    ## convolutional element
    ## we add padding at the end of the time win
    cnn_output = mx.sym.pad(data=conv_input,
                            mode="constant",
                            constant_value=0,
                            pad_width=(0, 0,
                                       0, 0,
                                       0, sz_filt - 1, 
                                       0, 0))
    cnn_output = mx.sym.Convolution(data=cnn_output,
                                    kernel=(sz_filt,
                                            input_feature_shape[2]),
                                    num_filter=n_filter)
    cnn_output = mx.sym.Activation(data=cnn_output, act_type='relu')
    cnn_output = mx.sym.reshape(mx.sym.transpose(data=cnn_output,
                                                 axes=(0, 2, 1, 3)),
                                shape=(0, 0, 0))
    cnn_output = mx.sym.Dropout(cnn_output, p=drop)

    ## recurrent element
    stacked_rnn_cells = mx.rnn.SequentialRNNCell()
    stacked_rnn_cells.add(mx.rnn.GRUCell(num_hidden=RNN_UNITS))
    outputs, _ = stacked_rnn_cells.unroll(length=win,
                                          inputs=cnn_output,
                                          merge_outputs=False)
    rnn_output = outputs[-1] # only take value from final unrolled cell for use later
    n_outputs = input_feature_shape[2]
    cnn_rnn_model = mx.sym.FullyConnected(data=rnn_output,
                                          num_hidden=n_outputs)

    ## ar element
    ar_outputs = []
    for i in list(range(input_feature_shape[2])):
        ar_series = mx.sym.slice_axis(data=X,
                                      axis=2,
                                      begin=i,
                                      end=i+1)
        fc_ar = mx.sym.FullyConnected(data=ar_series, num_hidden=1)
        ar_outputs.append(fc_ar)
    ar_model = mx.sym.concat(*ar_outputs, dim=1)

    output = cnn_rnn_model + ar_model
    loss_grad = mx.sym.LinearRegressionOutput(data=output, label=Y)
    return (loss_grad,
            [v.name for v in iter_train.provide_data],
            [v.name for v in iter_train.provide_label])


# %% [markdown]
# ## Training

# %%
################
## TRAINING ##
################

def train(symbol, iter_train, valid_iter, iter_test,
          data_names, label_names,
          save_dir, gpu):
    ## save training information/results 
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    printFile = open(os.path.join(SAVE_DIR, 'log.txt'), 'w')
    def print_to_file(msg):
        print(msg)
        print(msg, file = printFile, flush = True)
    ## print_to_file(args) ## preserve configuation to enable hyperparameter optimization
    ## archiving results header
    print_to_file('Epoch     Training Cor     Validation Cor')


    ## storing prior epoch's values to set an improvement threshold
    ## terminates early if progress slow
    buf     = RingBuffer(THRESHOLD_EPOCHS)
    old_val = None

    ## mxnet boilerplate
    ## defaults to 1 gpu of which index is 0
    ##devs = [mx.gpu(gpu)]
    devs   = mx.cpu()
    module = mx.mod.Module(symbol,
                           data_names=data_names,
                           label_names=label_names,
                           context=devs)
    module.bind(data_shapes=iter_train.provide_data,
                label_shapes=iter_train.provide_label)
    module.init_params(mx.initializer.Uniform(0.1))
    module.init_optimizer(optimizer='adam',
                          optimizer_params={'learning_rate': LR})

    ## training
    for epoch in range( N_EPOCHS):
        iter_train.reset()
        iter_val.reset()
        for batch in iter_train:
            module.forward(batch, is_train=True) # compute predictions
            module.backward()                    # compute gradients
            module.update()                      # update parameters

        ## training results
        train_pred  = module.predict(iter_train).asnumpy()
        train_label = iter_train.label[0][1].asnumpy()
        train_perf  = perf.write_eval(train_pred, train_label,
                                      save_dir, 'train', epoch)

        ## validation results
        val_pred  = module.predict(iter_val).asnumpy()
        val_label = iter_val.label[0][1].asnumpy()
        val_perf = perf.write_eval(val_pred, val_label,
                                   save_dir, 'valid', epoch)

        print_to_file('%d         %f       %f ' % (epoch, train_perf['COR'], val_perf['COR']))
        
        if epoch > 0:                                # if we don't yet have measures of improvement, skip
            buf.append(val_perf['COR'] - old_val) 
        if epoch > 2:                                # if we do have measures of improvement, check them
            vals = buf.get()
            # print(vals)
            # print(COR_THRESHOLD)
            vals = [v for v in vals if v != 0]
            if sum([v < COR_THRESHOLD for v in vals]) == len(vals):
                print_to_file('EARLY EXIT')
                break
        old_val = val_perf['COR']
                
    ## testing
    test_pred  = module.predict(iter_test).asnumpy()
    test_label = iter_test.label[0][1].asnumpy()
    test_perf = perf.write_eval(test_pred, test_label, save_dir, 'tst', epoch)
    print_to_file('\n TESTING PERFORMANCE')
    print_to_file(test_perf)

# %% [markdown]
# ## Run

# %%
# create data iterators
iter_train, iter_val, iter_test = prepare_iters(DATA_FILE, WIN, H, MODEL, BATCH_N)    

## prepare symbols
input_feature_shape = iter_train.provide_data[0][1]    
X                   = mx.sym.Variable(iter_train.provide_data[0].name)
Y                   = mx.sym.Variable(iter_train.provide_label[0].name)
    
# set up model
model_dict = {
    'fc_model'            : fc_model,
    'rnn_model'           : rnn_model,
    'cnn_model'           : cnn_model,
    'simple_lstnet_model' : simple_lstnet_model
    }

model = model_dict[MODEL]
    
symbol, data_names, label_names = model(iter_train,
                                        input_feature_shape, X, Y,
                                        WIN, SZ_FILT,
                                        N_FILT, DROP, SEASONAL_PERIOD)

## train 
train(symbol, iter_train, iter_val, iter_test, data_names, label_names, SAVE_DIR, GPU)

# %% [markdown]
# ## Exercise: load the results and evaluate the model performance

# %%
results_true = pd.read_csv("resultsDir/valid_label_24.csv", index_col=0)
results_pred  = pd.read_csv("resultsDir/valid_pred_24.csv", index_col=0)

# %%
results_true.head()

# %%

# %%
plt.scatter(results_true.iloc[:, 0], results_pred.iloc[:, 0])
pearsonr(results_true.iloc[:, 0], results_pred.iloc[:, 0])

# %%
plt.scatter(results_true.iloc[:, 25], results_pred.iloc[:, 25])
print(pearsonr(results_true.iloc[:,25], results_pred.iloc[:, 25]))
print(spearmanr(results_true.iloc[:,25], results_pred.iloc[:, 25]))

# %%
plt.scatter(results_true.iloc[:, 50], results_pred.iloc[:, 50])
print(pearsonr(results_true.iloc[:, 50], results_pred.iloc[:, 50]))

# %%
plt.plot(results_true.iloc[1800:2000, 50])
plt.plot(results_pred.iloc[1800:2000, 50] * 10)

# %%
plt.plot(results_true.iloc[1800:2000, 25])
plt.plot(results_pred.iloc[1800:2000, 25] * 10)

# %%
plt.hist(results_pred.iloc[1800:2000, 25])

# %% [markdown]
# ## Exercise: how does the model perform against the null model?

# %%
