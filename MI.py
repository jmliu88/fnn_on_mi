# coding: utf-8
from lasagne.regularization import *
import pdb

import numpy as np
from scipy import io

import sys
sys.path.append('./sequence_embedding/')
import utils

import lasagne
import theano

import theano.tensor as T
from sklearn.cross_validation import KFold
import os

## define a batch genetator which could be iterated
import random
class batch_generator():
    def __init__(self,x,y,m,MAX_BATCH_SIZE = 10,isShuffle = False):
        self.x = x
        self.y = y
        self.m = m
        self.data_index = range(len(x))
        self.rm_index = np.zeros_like(range(len(x)))
        if isShuffle:
            random.shuffle(self.data_index)
        self.size = MAX_BATCH_SIZE
    def get_batch(self,return_index = False):
        return_x = []
        return_y = []
        example_index = self.data_index[0]
        #print self.m[example_index]
        length = sum(self.m[example_index])
        #print len(self.index)
        rm_index = []
        for ind,i in enumerate(self.data_index):
            if sum(self.m[i]) == length:
                return_x.append(self.x[i][:length][:])
                return_y.append(self.y[i])
                rm_index.append(ind)
                if len(return_y) == self.size:
                    self.data_index = np.delete(self.data_index,rm_index)
                    return np.array(return_x,dtype=theano.config.floatX),np.array(return_y,dtype=theano.config.floatX)
        self.data_index = np.delete(self.data_index,rm_index)
        return np.array(return_x,dtype=theano.config.floatX),np.array(return_y,dtype=theano.config.floatX)
    def __iter__(self):
        return self
    def next(self):
        if len(self.data_index) == 0:
            raise StopIteration
        else:
            return self.get_batch()


## data formating
#directory = 'data/mat/fox_100x100_matlab.mat'
directory = sys.argv[1]#'data/mat/fox_100x100_matlab.mat'
D = io.loadmat(directory)
features0 = D['features'].todense()
## remove identical features
uniid = []
for i in range(features0.shape[1]):
    if len(np.unique(np.array(features0[:,i]))) == 1:
        uniid.append(i)
features = np.delete(features0,uniid,axis = 1)
## standardize all data (maybe flawed)
all_mean,all_std = utils.standardize(features)
features = (features - all_mean)/all_std

#features = features0
#pdb.set_trace()
labels = np.array(D['labels'].todense())[0]
bag_ids = D['bag_ids'][0]

MAX_LENGTH = max([list(bag_ids).count(iBag) for iBag in set(bag_ids)])
N_FEATURE_DIM = features.shape[1]

X = np.zeros((len(set(bag_ids)),MAX_LENGTH,N_FEATURE_DIM))
Y = np.zeros((len(set(bag_ids)),))
M = np.zeros((len(set(bag_ids)),MAX_LENGTH))

for iBag in set(bag_ids):
    instance_index = np.where(bag_ids == iBag)[0]
#    print instance_index[0]
#    print np.concatenate((features[instance_index],np.zeros((MAX_LENGTH-len(instance_index[0]),N_FEATURE_DIM))),axis = 0).shape
#    break
    X[iBag-1] = np.concatenate((features[instance_index],np.zeros((MAX_LENGTH-len(instance_index),N_FEATURE_DIM))),axis = 0).astype(theano.config.floatX)
    assert(len(set(labels[instance_index])) == 1)
    Y[iBag -1] = labels[instance_index[0]].astype(theano.config.floatX)
    Y[Y == -1] = 0
    M[iBag-1] = np.concatenate((np.ones(len(instance_index))
                                ,np.zeros((MAX_LENGTH-len(instance_index)))),axis = 0).astype(theano.config.floatX)

import csv
### train val test set
HIDDEN_SIZE = 100
DROPOUT_RATIO = 0
C = 50 # Factor of l2 norm
learning_rate = 0.001
for C in [100,50,20,10]:
    expDir = os.path.join('hidden_%d_dropout_%.1f_rmsprop_stand_all_feat_%.2f_l2norm_%.2flr_10_attention'%(HIDDEN_SIZE,DROPOUT_RATIO,C,learning_rate),os.path.basename(directory)+os.path.sep)
    if not os.path.isdir(expDir):
        os.makedirs(expDir)
        with open(os.path.join(expDir,'README'),'w') as fid:
            fid.write('learning rate = %f\n'%learning_rate)
            fid.write('dropout ratio = %f\n'%learning_rate)
            fid.write('penalty factor = %f\n'%C)
    result = np.zeros(shape=(10,10))
    for r in range(3):
        k=0
        kf = KFold(X.shape[0],10,True)
        for train_index,test_index in kf:
            input_shape = (None, MAX_LENGTH, N_FEATURE_DIM)
            # Construct network
            layer = lasagne.layers.InputLayer(shape=input_shape, name='Input')
            n_batch, n_seq, n_features = layer.input_var.shape
            # Store a dictionary which conveniently maps names to layers we will
            # need to access later
            layers = {'in': layer}
            # Add dense input layer
            layer = lasagne.layers.ReshapeLayer(
                layer, (n_batch*n_seq, input_shape[-1]), name='Reshape 1')
            layer = lasagne.layers.DenseLayer(
                layer, HIDDEN_SIZE, W=lasagne.init.HeNormal(), name='Input dense',
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
            layer = lasagne.layers.ReshapeLayer(
                layer, (n_batch, n_seq, HIDDEN_SIZE), name='Reshape 2')
            # Add the layer to aggregate over time steps
                # We must force He initialization because Lasagne doesn't like
                # 1-dim shapes in He and Glorot initializers
            layer = utils.AttentionLayer(
                    layer,10,
                    W=lasagne.init.Normal(1./np.sqrt(layer.output_shape[-1])),
                    name='Attention')
            N_LAYERS = 1
            for _ in range(N_LAYERS):
                layer = lasagne.layers.DenseLayer(
                    layer, HIDDEN_SIZE, W=lasagne.init.HeNormal(), name='Out dense 1',
                    nonlinearity=lasagne.nonlinearities.leaky_rectify)
                layer = lasagne.layers.DropoutLayer(layer, p=DROPOUT_RATIO)
            # Add final dense layer, whose bias is initialized to the target mean
            layer = lasagne.layers.DenseLayer(
                layer, 1, W=lasagne.init.HeNormal(), name='Out dense 3',
                nonlinearity=lasagne.nonlinearities.sigmoid)
            layer = lasagne.layers.ReshapeLayer(
                layer, (-1,))
            # Keep track of the final layer
            layers['out'] = layer
            #l2_norm = regularize_layer_params(layer,l2)
            l2_norm = regularize_layer_params(lasagne.layers.get_all_layers(layers['out']),l2)

            # Symbolic variable for target values
            target = T.vector('target')
            # Retrive the symbolic expression for the network
            network_output = lasagne.layers.get_output(layers['out'])
            # Create a symbolic function for the network cost
            cost = T.mean(lasagne.objectives.binary_crossentropy(network_output,target))
            cost = cost + C*l2_norm
            #cost = T.mean((network_output - target)**2)
            # Retrieve all network parameters
            all_params = lasagne.layers.get_all_params(layers['out'])
            # Compute updates
            updates = lasagne.updates.adam(cost, all_params, learning_rate)
            # Compile training function
            train = theano.function([layers['in'].input_var, target],
                                    cost, updates=updates)

            # Accuracy is defined as binary accuracy
            compute_cost = theano.function([layers['in'].input_var, target],
                                    cost)
            accuracy = T.sum(lasagne.objectives.binary_accuracy(network_output, target))
            compute_accuracy = theano.function(
                [layers['in'].input_var, target], accuracy)
            #print 'Model built.'


            X_train_all, X_test = X[train_index], X[test_index]
            y_train_all, y_test = Y[train_index], Y[test_index]
            m_train_all, m_test = M[train_index], M[test_index]
            kf_val = KFold(X_train_all.shape[0],10,True)
            for train_ind,val_ind in kf_val:
                X_train, X_val = X_train_all[train_ind], X_train_all[val_ind]
                y_train, y_val = y_train_all[train_ind], y_train_all[val_ind]
                m_train, m_val = m_train_all[train_ind], m_train_all[val_ind]
                break
            ## standardize three sets
#        x_tr_mean,x_tr_std = utils.standardize(X_train)
#
#        X_train = (X_train-x_tr_mean)/x_tr_std
#        X_val = (X_val-x_tr_mean)/x_tr_std
#        X_test = (X_test-x_tr_mean)/x_tr_std
#        print X_train
#        pdb.set_trace()

            MAX_EPOCH = 500
            NO_BEST = 10
            train_acc = np.array([])
            train_cost = np.array([])
            test_acc = np.array([])
            test_cost = np.array([])
            val_acc = np.array([])
            val_cost = np.array([])
            early_stop = False

            for iEpoch in range(MAX_EPOCH):
                b = batch_generator(X_train,y_train,m_train)
                trac = 0
                trco = 0
                for x_b,y_b in b:
                    #print x_b.shape,y_b.shape
                    train(x_b,y_b)
                    b_cost = compute_cost(x_b,y_b)
                    trco += b_cost
                    trac +=  compute_accuracy(x_b,y_b)
                if any([not np.isfinite(b_cost),
                            any([not np.all(np.isfinite(p.get_value()))
                            for p in all_params])]):
        #                logger.info('####### Non-finite values found, aborting')
                    print '####### Non-finite values found, aborting'
                    break
                train_acc = np.append(train_acc, trac/X_train.shape[0]) #compute_accuracy(x_b,y_b)
                train_cost = np.append(train_cost,trco/X_train.shape[0])

                vaco = 0
                vaac = 0
                bv = batch_generator(X_val,y_val,m_val)
                for xv,yv in bv:
                    vaco += compute_cost(xv,yv)
                    vaac += compute_accuracy(xv,yv)
                val_cost = np.append(val_cost,vaco)
                val_acc = np.append(val_acc,vaac)

                teac = 0
                teco = 0
                bt = batch_generator(X_test,y_test,m_test)
                for xt,yt in bt:
                    teac+= compute_accuracy(xt,yt)
                    teco+= compute_cost(xt,yt)
                test_acc = np.append(test_acc,teac)
                test_cost = np.append(test_cost,teac)

                if iEpoch > NO_BEST:
                    early_stop = True
                    last_val = val_cost[-NO_BEST: ]
                    for i,v in enumerate(last_val[:-2]):
                        if last_val[i] >= last_val[i+1]:
                            early_stop = False
                            break
                if early_stop:
                    #print "early stoping, last %s validation costs are: "%NO_BEST + ','.join([str(tmp) for tmp in last_val])
                    break
            best_model = np.argmax(val_acc)

            #print train_acc
            #print train_cost
            #print val_cost
            #print test_acc
            #print 'Reach maxmal validation acc at %dth iteration'%best_model
            #print 'train_cost = %f'%train_cost[best_model]
            #print 'val_cost = %f'%val_cost[best_model]
            #print 'test_acc = %f'%test_acc[best_model]
            result[r][k] = test_acc[best_model]
            print "%d times, %d folder finished, test acc is %f"%(r,k,test_acc[best_model]/X_test.shape[0])
            k=k+1

    print np.mean(result[:])
    with open(os.path.join( expDir, 'result_%f.csv'%(np.mean(result[:])/X_test.shape[0]) ),'w') as fid:
        writer = csv.writer(fid)
        writer.writerows(result)



# In[72]:

#get_ipython().system(u'cat fox_100x100_matlab.mat/result10.420000.csv')



