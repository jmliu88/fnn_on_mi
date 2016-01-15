
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

