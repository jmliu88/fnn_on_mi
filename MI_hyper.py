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
#    for POWER in [0.8,0.9,1]:
#    for INPUT_NOISE in [0.5,1,2,0.05,0.1,0.01]:
#        for C in [0.0005,0.001,0,0.0001]:
#            for REG in [l1,l2]:
#    ATTENTION_HIDDEN = 10
#    HIDDEN_SIZE = 100
#    N_LAYERS = 1

def main_test(POWER, INPUT_NOISE, C, REG, ATTENTION_HIDDEN, HIDDEN_SIZE, N_LAYERS):
    return random.randint()
def main(POWER=1, INPUT_NOISE=False, C=0, REG=l1, ATTENTION_HIDDEN=10, HIDDEN_SIZE=100, N_LAYERS=3):
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

    from sklearn.decomposition import PCA
    pca = PCA()
    pca.fit(features)
    loading = pca.explained_variance_ratio_

    n_components = len(loading)
    for p in range(len(loading)):
        if sum(loading[:p])>POWER:
            n_components = p
            break
    features = pca.transform(features)[:,:p]
#pdb.set_trace()

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
    DROPOUT_RATIO = 0
    learning_rate = 0.001
    R = 3
    s = '%s'%REG
    s = s.split()[1] ## s is the name of regularization
    if os.path.isdir(sys.argv[2]):
        saveDir = sys.argv[2]
    else:
        saveDir = 'exp_spearmint/'
    expDir = os.path.join(saveDir,os.path.basename(directory),'PCA%.1f_innoise_%f_%snorm_%f_attention_%d_hidden_%d_layers_%d'%(POWER, INPUT_NOISE,  s, C, ATTENTION_HIDDEN, HIDDEN_SIZE, N_LAYERS)+os.path.sep)
    if not os.path.isdir(expDir):
        os.makedirs(expDir)
        with open(os.path.join(expDir,'README'),'w') as fid:
            fid.write('learning rate = %f\n'%learning_rate)
            fid.write('dropout ratio = %f\n'%learning_rate)
            fid.write('penalty factor = %f\n'%C)
    result = np.zeros(shape=(R,10))
    for r in range(R):
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
            layer = lasagne.layers.GaussianNoiseLayer(layer,INPUT_NOISE)
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
            layer1 = utils.AttentionLayer(
                    layer,ATTENTION_HIDDEN,
                    W=lasagne.init.Normal(1./np.sqrt(layer.output_shape[-1])),
                    name='Attention')
            layer2 = utils.AttentionLayer(
                    layer,ATTENTION_HIDDEN,
                    W=lasagne.init.Normal(1./np.sqrt(layer.output_shape[-1])),
                    name='Attention')
            layer = lasagne.layers.MergeLayer([layer1,layer2])
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
            #l_norm = regularize_layer_params(layer,l1)
            l_norm = regularize_layer_params(lasagne.layers.get_all_layers(layers['out']),REG)

            # Symbolic variable for target values
            target = T.vector('target')
            # Retrive the symbolic expression for the network
            network_output = lasagne.layers.get_output(layers['out'],deterministic=True)
            # Create a symbolic function for the network cost
            cost = T.mean(lasagne.objectives.binary_crossentropy(network_output,target))
            # try Hinge loss
            #cost = T.mean(lasagne.objectives.binary_hinge_loss(network_output,target))
            cost = cost + C*l_norm
            #cost = T.mean((network_output - target)**2)
            # Retrieve all network parameters
            all_params = lasagne.layers.get_all_params(layers['out'])
            # Compute updates
            updates = lasagne.updates.rmsprop(cost, all_params, learning_rate )
            # Compile training function
            train = theano.function([layers['in'].input_var, target],
                                    cost, updates=updates)

            # Accuracy is defined as binary accuracy
            compute_cost = theano.function([layers['in'].input_var, target],
                                    cost,)
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
            best_model = np.argmin(val_cost)

            #print train_acc
            #print train_cost
            #print val_cost
            #print test_acc
            #print 'Reach maxmal validation acc at %dth iteration'%best_model
            #print 'train_cost = %f'%train_cost[best_model]
            #print 'val_cost = %f'%val_cost[best_model]
            #print 'test_acc = %f'%test_acc[best_model]
            result[r][k] = test_acc[best_model]/X_test.shape[0]
            print "%d times, %d folder finished, test acc is %f"%(r,k,test_acc[best_model]/X_test.shape[0])
            #pdb.set_trace()
            with open(os.path.join( expDir, 'val_cost_r%d_k%d.csv'%(r,k) ),'w') as fid:
                writer = csv.writer(fid)
                writer.writerows([val_cost])
            with open(os.path.join( expDir, 'val_acc_r%d_k%d.csv'%(r,k) ),'w') as fid:
                writer = csv.writer(fid)
                writer.writerows([val_acc])
            with open(os.path.join( expDir, 'test_cost_r%d_k%d.csv'%(r,k) ),'w') as fid:
                writer = csv.writer(fid)
                writer.writerows([test_cost])
            with open(os.path.join( expDir, 'test_acc_r%d_k%d.csv'%(r,k) ),'w') as fid:
                writer = csv.writer(fid)
                writer.writerows([test_acc])
            k=k+1

    print np.mean(result[:])
    with open(os.path.join( expDir, 'result_%f_%f.csv'%(np.mean(result[:]),np.std(result[:])) ),'w') as fid:
        writer = csv.writer(fid)
        writer.writerows(result)
    return -np.mean(result[:])

def main_hyper():
    import simple_spearmint
    parameter_space = {'POWER'      : {'type': 'float', 'min': 0.7, 'max': 1},
                    'INPUT_NOISE': {'type': 'float', 'min': 0.01, 'max': 1},
                    'C'          : {'type': 'float', 'min': 0.0001, 'max': 1},
                    'REG'        : {'type': 'enum', 'options': [l1, l2]},
                    'ATTENTION_HIDDEN': {'type': 'int', 'min':1, 'max': 50},
                    'HIDDEN_SIZE': {'type': 'int', 'min':10, 'max':200},
                    'N_LAYERS': {'type': 'int', 'min':1, 'max':8}}
    ss = simple_spearmint.SimpleSpearmint(parameter_space)
    for n in xrange(5):
        # Get random parameter settings
        suggestion = ss.suggest_random()
        # Retrieve an objective value for these parameters
        value = main(suggestion['POWER'], # How much of the variance that PCA should explain.
                        suggestion['INPUT_NOISE'], # input noise after PCA
                        suggestion['C'], # penalty factor
                        suggestion['REG'], # l1norm or l2norm
                        suggestion['ATTENTION_HIDDEN'], # how many hidden units in the attention layer
                        suggestion['HIDDEN_SIZE'], # hidden size for the classifier fully connected layers
                        suggestion['N_LAYERS'])# number of layers for the classifier fully connected layers
        print "Random trial {}: {} -> {}".format(n + 1, suggestion, value)
        # Update the optimizer on the result
        ss.update(suggestion, value)
    for n in xrange(100):
        # Get a suggestion from the optimizer
        suggestion = ss.suggest()
        # Get an objective value; the ** syntax is equivalent to
        # the call to objective above
        value = main(**suggestion)
        print "GP trial {}: {} -> {}".format(n + 1, suggestion, value)
        # Update the optimizer on the result
        ss.update(suggestion, value)
    best_parameters, best_objective = ss.get_best_parameters()
    print "Best parameters {} for objective {}".format(
        best_parameters, best_objective)
if __name__ == '__main__':
    main()
    #main_hyper()
