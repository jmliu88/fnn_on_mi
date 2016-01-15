
# In[1]:

from __future__ import print_function

import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T

import lasagne


# Out[1]:

#     Using gpu device 0: GeForce GTX 960 (CNMeM is disabled)
#

# In[2]:

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
NUM_EPOCHS = 100
# Number of feature dimentions
N_FEATURE_DIM = 39
MAX_LENGTH = 5000


# In[3]:

class Model():
    def __init__(self,label_index):
        self.n_class = len(set(label_index))
        self.label_index = label_index
        pass
    def compile(self):
        N_LSTM = 256
        N_DENSE = 10
        # First, we build the network, starting with an input layer
        # Recurrent layers expect input of shape
        # (batch size, max sequence length, number of features)
        l_in = lasagne.layers.InputLayer(shape=(N_BATCH, MAX_LENGTH, N_FEATURE_DIM))
        l_mask = lasagne.layers.InputLayer(shape=(N_BATCH, MAX_LENGTH))

        l_forward = lasagne.layers.LSTMLayer(l_in, N_LSTM, mask_input=l_mask,only_return_final=True)

        l_dense = lasagne.layers.DenseLayer(l_forward,num_units=N_DENSE)

        self.l_out = lasagne.layers.DenseLayer(
            l_dense, num_units=self.n_class, nonlinearity=lasagne.nonlinearities.softmax)

        target_values = T.ivector('target_output')

        network_output = lasagne.layers.get_output(self.l_out)

        cost = lasagne.objectives.categorical_crossentropy(network_output,target_values)
        cost = cost.mean()
        # Retrieve all parameters from the network
        all_params = lasagne.layers.get_all_params(self.l_out)
        # Compute SGD updates for training
        updates = lasagne.updates.adagrad(cost, all_params, LEARNING_RATE)

        self.train = theano.function([l_in.input_var, target_values, l_mask.input_var],
                                cost, updates=updates,allow_input_downcast=True)

        self.compute_cost = theano.function(
            [l_in.input_var, target_values, l_mask.input_var], cost,allow_input_downcast=True)

        self.get_output = theano.function(
            [l_in.input_var, l_mask.input_var], lasagne.layers.get_output(self.l_out, deterministic=True),allow_input_downcast=True)
    def load(self,model_file):
        assert(os.path.splitext(model_file)[-1] == '.npz')
        # load model
        with np.load(model_file) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(self.l_out, param_values)
        print("Building network ... DONE")
        return self.l_out,self.get_output
    def predict(self,x,m):
        decision_value = self.get_output(x,m)
        print(decision_value)
        val_predictions = np.argmax(decision_value, axis=1)
        final_predictions = [self.label_index[t] for t in val_predictions]
        return val_predictions,final_predictions
    def update(self,X_batch,Y_batch,m_batch):
        self.train(X_batch,Y_batch,m_batch)
    def compute_cost(self,X_batch,Y_batch,m_batch):
        self.compute_cost(X_batch,Y_batch,m_batch)
    def save(self,savename):
        np.savez(savename, *lasagne.layers.get_all_param_values(self.l_out))


# In[4]:

def format_data(func):
    def wrapper(*args, **kw):
        mfcc = np.array(func(*args,**kw))[2000:]
        X = np.zeros(shape=(1, MAX_LENGTH, mfcc.shape[1]))
        if mfcc.shape[0] > MAX_LENGTH:
            X[0,:,:] = mfcc[:MAX_LENGTH]
            return X, np.ones([1,MAX_LENGTH])
        else:
            X[0,0:mfcc.shape[0],:] = mfcc[:MAX_LENGTH]
            return X, np.array([1]*mfcc.shape[0]+[0]*(MAX_LENGTH-mfcc.shape[0]))
    return wrapper


# In[5]:

@format_data
def readHTKfeat(dataPath):
    data = np.array([[]])
    flagDataStart = 0
    nLine = 0
    i = -1
    for line in open(dataPath,'r'):
        if line.find('END') != -1:
            return data
        if flagDataStart:
            if nLine == 0:
                dataLine = np.array([float(x) for x in line.split()[1:]])
                nLine = nLine + 1
                i = i+1
                continue
            elif nLine in [1,2]:
                dataLine = np.concatenate((dataLine,[float(x) for x in line.split()]))
                nLine = nLine + 1
                continue
            elif nLine == 3:
                dataLine = np.concatenate((dataLine,[float(x) for x in line.split()]))
                data[i,:] = dataLine
                nLine = 0
                continue

        if line.find('------------------------------------ Samples: 0->-1 ------------------------------------') != -1:
            flagDataStart = 1

        if line.find('Num Samples') != -1:
            nSamples = int(line.split()[2])
            data = np.ndarray(shape=(nSamples,N_FEATURE_DIM))
            flagDataStart= 0

    return data


# In[6]:

def read_from_list(f_list):
    for iFile in f_list:
        #print(iFile)
        f = h5py.File(iFile.strip())
        if 'X' not in locals():
            X = f['data'][:,2000:5000:,:]
            Y = f['label'][:]
            mask = f['mask'][:,2000:5000:]
        else:
            X = np.concatenate((X,f['data'][:,2000:5000:,:]),axis=0).astype('float32')
            Y = np.concatenate((Y,f['label'][:]),axis=0).astype('float32')
            mask = np.concatenate((mask,f['mask'][:,2000:5000:]),axis=0).astype('float32')
        f.close()
    return (X,Y,mask)


# In[9]:

if __name__ == '__main__':
    import time
    import h5py
    l_out,get_output = build_nn('models/trained_15_1.81667185837_1.82911217213_0.141463414634.npz')
    lab = ['FN','BS','XQ','KX','JJ','HP','nobark']
    testlist = 'exp/test.list'
    with open(testlist,'r') as fid:
        testf = fid.readlines()

    x,y,m = read_from_list(testf)
#    print(testf[0:1])
#    x = x[0:1]
#    y = y[0:1]
#    m = m[0:1]
#    testfile = '../data/MFCC/MFCC_E_D_A_txt/Chihuahua_male_1_holding_scary_20150928_123719.txt'
    #mfcc
    stime = time.time()
#    x,m = readHTKfeat(testfile)
    print(time.time() - stime)
    print(x.shape,m.shape)
    decision_value = get_output(x,m)
    print(decision_value)
    val_predictions = np.argmax(decision_value, axis=1)
    print(time.time()-stime)
    final_prediction = [lab[t] for t in val_predictions]
    print(final_prediction)
    print([lab[t] for t in y.astype('int32')])


# Out[9]:

#     Building network ...
#     Building network ... DONE
#     2.14576721191e-06
#     (205, 3000, 39) (205, 3000)
#     [[ 0.14534315  0.11807641  0.16955249  0.2546348   0.12220483  0.19018836]
#      [ 0.14865403  0.13375641  0.17538147  0.25812808  0.13354664  0.15053336]
#      [ 0.24792486  0.07962665  0.22670573  0.12711224  0.12786676  0.19076377]
#      ...,
#      [ 0.14621386  0.11843522  0.11727074  0.09500909  0.35057852  0.17249261]
#      [ 0.24302116  0.06831074  0.10927071  0.11626498  0.29634595  0.1667864 ]
#      [ 0.1600578   0.13143404  0.16640334  0.23065552  0.15335669  0.15809269]]
#     0.867572069168
#     ['KX', 'KX', 'FN', 'JJ', 'FN', 'HP', 'HP', 'KX', 'FN', 'KX', 'JJ', 'JJ', 'XQ', 'KX', 'XQ', 'KX', 'JJ', 'KX', 'HP', 'JJ', 'KX', 'JJ', 'HP', 'KX', 'KX', 'FN', 'FN', 'XQ', 'KX', 'JJ', 'FN', 'FN', 'JJ', 'FN', 'KX', 'FN', 'KX', 'FN', 'HP', 'FN', 'JJ', 'KX', 'FN', 'JJ', 'XQ', 'XQ', 'JJ', 'FN', 'JJ', 'FN', 'JJ', 'FN', 'XQ', 'KX', 'JJ', 'XQ', 'KX', 'JJ', 'FN', 'JJ', 'FN', 'JJ', 'FN', 'KX', 'KX', 'FN', 'FN', 'FN', 'FN', 'KX', 'FN', 'HP', 'HP', 'FN', 'FN', 'KX', 'JJ', 'HP', 'HP', 'JJ', 'KX', 'HP', 'KX', 'XQ', 'JJ', 'FN', 'FN', 'FN', 'JJ', 'FN', 'JJ', 'FN', 'FN', 'FN', 'JJ', 'KX', 'FN', 'HP', 'FN', 'JJ', 'XQ', 'FN', 'JJ', 'KX', 'JJ', 'FN', 'JJ', 'JJ', 'FN', 'JJ', 'JJ', 'FN', 'KX', 'JJ', 'JJ', 'FN', 'JJ', 'HP', 'JJ', 'XQ', 'KX', 'FN', 'XQ', 'JJ', 'FN', 'HP', 'HP', 'FN', 'JJ', 'FN', 'FN', 'KX', 'JJ', 'JJ', 'FN', 'JJ', 'XQ', 'XQ', 'KX', 'JJ', 'KX', 'JJ', 'JJ', 'FN', 'FN', 'JJ', 'KX', 'KX', 'FN', 'FN', 'KX', 'JJ', 'KX', 'JJ', 'JJ', 'FN', 'FN', 'XQ', 'HP', 'JJ', 'HP', 'FN', 'JJ', 'KX', 'FN', 'JJ', 'KX', 'JJ', 'KX', 'FN', 'FN', 'XQ', 'FN', 'JJ', 'FN', 'XQ', 'FN', 'HP', 'FN', 'KX', 'KX', 'FN', 'JJ', 'FN', 'KX', 'KX', 'KX', 'FN', 'FN', 'FN', 'JJ', 'JJ', 'KX', 'FN', 'KX', 'JJ', 'KX', 'FN', 'JJ', 'JJ', 'FN', 'HP', 'JJ', 'JJ', 'KX']
#     ['FN', 'FN', 'JJ', 'KX', 'FN', 'KX', 'FN', 'KX', 'FN', 'KX', 'JJ', 'XQ', 'KX', 'XQ', 'XQ', 'FN', 'JJ', 'JJ', 'BS', 'BS', 'HP', 'JJ', 'FN', 'BS', 'JJ', 'BS', 'FN', 'KX', 'KX', 'JJ', 'JJ', 'FN', 'XQ', 'FN', 'KX', 'XQ', 'FN', 'KX', 'BS', 'FN', 'KX', 'BS', 'JJ', 'BS', 'JJ', 'XQ', 'XQ', 'XQ', 'JJ', 'KX', 'JJ', 'HP', 'FN', 'FN', 'HP', 'HP', 'KX', 'XQ', 'BS', 'BS', 'XQ', 'FN', 'HP', 'KX', 'FN', 'JJ', 'XQ', 'FN', 'JJ', 'XQ', 'KX', 'JJ', 'XQ', 'JJ', 'XQ', 'JJ', 'FN', 'FN', 'FN', 'FN', 'BS', 'XQ', 'XQ', 'JJ', 'JJ', 'XQ', 'JJ', 'FN', 'HP', 'XQ', 'XQ', 'XQ', 'BS', 'BS', 'KX', 'HP', 'JJ', 'JJ', 'FN', 'JJ', 'KX', 'KX', 'FN', 'KX', 'HP', 'XQ', 'FN', 'KX', 'BS', 'HP', 'KX', 'JJ', 'JJ', 'JJ', 'HP', 'JJ', 'XQ', 'FN', 'JJ', 'KX', 'JJ', 'BS', 'XQ', 'FN', 'JJ', 'JJ', 'XQ', 'KX', 'XQ', 'FN', 'FN', 'JJ', 'BS', 'XQ', 'JJ', 'BS', 'XQ', 'KX', 'BS', 'FN', 'BS', 'JJ', 'BS', 'KX', 'XQ', 'KX', 'XQ', 'XQ', 'JJ', 'XQ', 'FN', 'HP', 'FN', 'XQ', 'JJ', 'XQ', 'XQ', 'XQ', 'BS', 'JJ', 'JJ', 'XQ', 'XQ', 'JJ', 'JJ', 'KX', 'KX', 'XQ', 'KX', 'XQ', 'BS', 'BS', 'JJ', 'HP', 'BS', 'BS', 'FN', 'JJ', 'XQ', 'BS', 'HP', 'BS', 'XQ', 'JJ', 'BS', 'XQ', 'XQ', 'XQ', 'KX', 'BS', 'KX', 'FN', 'HP', 'JJ', 'KX', 'JJ', 'XQ', 'BS', 'JJ', 'XQ', 'XQ', 'KX', 'KX', 'FN', 'XQ']
#
