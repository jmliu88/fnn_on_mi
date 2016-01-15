
# In[1]:

from __future__ import print_function

import sys
import os
import time

import numpy as np

import random

from feature_operation import *

# Out[1]:

#     Using gpu device 0: GeForce GTX 960 (CNMeM is disabled)
#

# In[2]:


# In[3]:

## Setting

# Number of units in the hidden (recurrent) layer
N_HIDDEN = 10
# Number of training sequences in each batch
N_BATCH = 10
# Optimization learning rate
LEARNING_RATE = .001
# All gradients above this will be clipped
GRAD_CLIP = 100
# How often should we check the output?
EPOCH_SIZE = 1
# Number of epochs to train the net
NUM_EPOCHS = 1000
# Number of feature dimentions
N_FEATURE_DIM = 39

MAX_LENGTH = 5000

# In[4]:

# In[5]:

## Load Feature


# In[6]:

def early_stop(valArray,NOT_BETTER_NUM = 10):
    valArray = valArray[-NOT_BETTER_NUM:]
    if len(valArray) < NOT_BETTER_NUM:
        return False
    else:
        length = len(valArray)
        for i,n in enumerate(valArray[length-NOT_BETTER_NUM:-2]):
            if valArray[i]>valArray[i+1]:
                return False
        return True
class data_generator_from_memory():
    def __init__(self,x,y,m,n_batch=N_BATCH,isShuffle = True):
        import random
        # data
        self.x=x
        self.y=y
        self.m=m
        self.n_batch = n_batch
        # index
        self.index = range(len(y))
        if isShuffle:
            random.shuffle(self.index)
        self.used_index = []
    def get_batch(self):
        self.used_index = self.used_index + self.index[:self.n_batch]
        minibatch = (self.x[self.index[:self.n_batch]],self.y[self.index[:self.n_batch]],self.m[self.index[:self.n_batch]])
        self.index = self.index[self.n_batch:]
        return minibatch


# In[7]:

class data_generator_from_h5():
    # use this function to read minibatch from hard disk, in order to reduce memory usage
    def __init__(self,file_list,n_batch=N_BATCH,isShuffle = True):
        import random
        # data
        self.list = file_list
        self.n_batch = n_batch
        # index
        self.index = range(len(file_list))
        if isShuffle:
            random.shuffle(self.index)
        self.used_index = []

    def get_batch(self):
        self.used_index = self.used_index + self.index[:self.n_batch]
        minibatch = read_from_list(self.list[self.index[:self.n_batch]])
        self.index = self.index[self.n_batch:]
        return minibatch
class data_generator_from_htk(data_generator_from_h5):
    def get_batch(self):
        self.used_index = self.used_index + self.index[:self.n_batch]
        minibatch = read_htk_list(self.list[self.index[:self.n_batch]])
        self.index = self.index[self.n_batch:]
        return minibatch


# In[8]:

def save_list(list_data,savename):
    with open(savename,'w') as fid:
        for i in list_data:
            fid.write(i+'\n')

# In[9]:
if __name__ == '__main__':
# shuffle dataset
    from progressbar import ProgressBar
    print('Preparing data...')
    mfcDir = '/home/james/data/mfcc_all'
    dataList = np.array([os.path.join(mfcDir ,ifile) for ifile in os.listdir(mfcDir )])
    N_SEQ = len(dataList)
    index = range(N_SEQ)
    random.shuffle(index)

    ## split into train val test
    trainPortion,valPortion = [int(i*N_SEQ) for i in [0.95,0.98]]

    list_train = dataList[index[:trainPortion]]
    save_list(list_train,'exp/train.list')
    list_val = dataList[index[trainPortion:valPortion]]
    save_list(list_val,'exp/val.list')
#    X_val,Y_val,mask_val = read_htk_list(list_val)
    list_test = dataList[index[valPortion:]]
    save_list(list_test,'exp/test.list')
#    X_test,Y_test,mask_test = read_htk_list(list_test)
    ## init model
    import model
    mdl = model.Model(label_index)
    mdl.compile()
    ## train model
    print("Training ...")
    print(time.strftime('%X %x %Z'))
    earlyStop=0
    costTrainArray = []
    costValArray = []
    hitTestArray = []

    p = ProgressBar()

    try:
        EPOCH_SIZE = len(list_train)/N_BATCH
        #EPOCH_SIZE = 1#len(list_train)/N_BATCH
        for epoch in range(NUM_EPOCHS):
            data = data_generator_from_htk(list_train,isShuffle=True)
            data_val = data_generator_from_htk(list_val,isShuffle=True)
            data_test = data_generator_from_htk(list_test,isShuffle=True)
            cost_train = 0
            p.start()
            for _ in range(EPOCH_SIZE):
                X_batch, Y_batch, m_batch = data.get_batch()
    #            print(X_batch.shape,Y_batch.shape,m_batch.shape)
                mdl.update(X_batch, Y_batch, m_batch)
                cost_batch = mdl.compute_cost(X_batch, Y_batch, m_batch)
                cost_train = cost_train + cost_batch
                p.update(int(float(_)/EPOCH_SIZE*100))
            #    print("{} minibatch cost = {}".format(_,cost_batch))
            p.finish()
            cost_train = cost_train/EPOCH_SIZE
            cost_val = 0
            for val_epoch in range(len(list_val)/N_BATCH):
                x_v,y_v,m_v = data_val.get_batch()
                cost_val = cost_val + mdl.compute_cost(x_v,y_v,m_v)/N_BATCH
            hit = 0
            for test_epoch in range(len(list_test)/N_BATCH):
                x_t,y_t,m_t = data_test.get_batch()
                pred = mdl.predict(x_t,m_t)[0]
                hit = hit + np.sum(pred==y_t)
            acc = hit/float(len(list_test))

            print("Epoch {} train cost = {}".format(epoch, cost_train))
            print("Epoch {} validation cost = {}".format(epoch, cost_val))
            print("Epoch {} correctly predicted {} out of {}".format(epoch, hit,len(list_test)))
            print(time.strftime('%X %x %Z'))
            mdl.save('models_8class/trained_{}_{}_{}_{}.npz'.format(epoch,cost_train,cost_val,acc))
            costTrainArray.append(cost_train)
            costValArray.append(cost_val)
            hitTestArray.append(hit)

            earlyStop = early_stop(costValArray,50)
            if earlyStop:
                break
    except KeyboardInterrupt:
        pass


#    # In[10]:
#
#    print("Building network ...")
#    N_LSTM = 10
#    N_DENSE = 10
#    # First, we build the network, starting with an input layer
#    # Recurrent layers expect input of shape
#    # (batch size, max sequence length, number of features)
#    l_in = lasagne.layers.InputLayer(shape=(N_BATCH, MAX_LENGTH, N_FEATURE_DIM))
#    # The network also needs a way to provide a mask for each sequence.  We'll
#    # use a separate input layer for that.  Since the mask only determines
#    # which indices are part of the sequence for each batch entry, they are
#    # supplied as matrices of dimensionality (N_BATCH, MAX_LENGTH)
#    l_mask = lasagne.layers.InputLayer(shape=(N_BATCH, MAX_LENGTH))
#    # Setting a value for grad_clipping will clip the gradients in the layer
#    # Setting only_return_final=True makes the layers only return their output
#    # for the final time step, which is all we need for this task
#
#    #l_forward1 = lasagne.layers.LSTMLayer(l_in, N_LSTM, mask_input=l_mask,only_return_final=False)
#    #l_forward2 = lasagne.layers.LSTMLayer(l_forward1, N_LSTM, mask_input=l_mask,only_return_final=False)
#    #l_forward3 = lasagne.layers.LSTMLayer(l_forward2, N_LSTM, mask_input=l_mask,only_return_final=False)
#    l_forward = lasagne.layers.LSTMLayer(l_in, N_LSTM, mask_input=l_mask,only_return_final=True)
#    #l_forward = lasagne.layers.RecurrentLayer(
#    #    l_in, N_HIDDEN, mask_input=l_mask, grad_clipping=GRAD_CLIP,
#    #    W_in_to_hid=lasagne.init.HeUniform(),
#    #    W_hid_to_hid=lasagne.init.HeUniform(),
#    #    nonlinearity=lasagne.nonlinearities.tanh, only_return_final=True)
#
#    #    l_backward = lasagne.layers.RecurrentLayer(
#    #        l_in, N_HIDDEN, mask_input=l_mask, grad_clipping=GRAD_CLIP,
#    #        W_in_to_hid=lasagne.init.HeUniform(),
#    #        W_hid_to_hid=lasagne.init.HeUniform(),
#    #        nonlinearity=lasagne.nonlinearities.tanh,
#    #        only_return_final=True, backwards=True)
#    #    # Now, we'll concatenate the outputs to combine them.
#    #    l_concat = lasagne.layers.ConcatLayer([l_forward, l_backward])
#    # Our output layer is a simple dense connection, with 1 output unit
#    #l_dropout = lasagne.layers.dropout(l_forward)
#    l_dense = lasagne.layers.DenseLayer(l_forward,num_units=N_DENSE)
#
#    l_out = lasagne.layers.DenseLayer(
#        l_dense, num_units=6, nonlinearity=lasagne.nonlinearities.softmax)
#
#    target_values = T.ivector('target_output')
#
#    # lasagne.layers.get_output produces a variable for the output of the net
#    network_output = lasagne.layers.get_output(l_out)
#
#    # Our cost will be mean-squared error
#    #cost = T.mean((predicted_values - target_values)**2)
#    cost = lasagne.objectives.categorical_crossentropy(network_output,target_values)
#    cost = cost.mean()
#    # Retrieve all parameters from the network
#    all_params = lasagne.layers.get_all_params(l_out)
#    # Compute SGD updates for training
#    print("Computing updates ...")
#    updates = lasagne.updates.adagrad(cost, all_params, LEARNING_RATE)
#    # Theano functions for training and computing cost
#    print("Compiling functions ...")
#    train = theano.function([l_in.input_var, target_values, l_mask.input_var],
#                            cost, updates=updates,allow_input_downcast=True)
#    compute_cost = theano.function(
#        [l_in.input_var, target_values, l_mask.input_var], cost,allow_input_downcast=True)
#    get_output = theano.function(
#        [l_in.input_var, l_mask.input_var], lasagne.layers.get_output(l_out, deterministic=True),allow_input_downcast=True)
#    print("Building network ... DONE")


    # We'll use this "validation set" to periodically check progress
