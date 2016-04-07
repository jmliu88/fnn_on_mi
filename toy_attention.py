import lasagne
import theano
import theano.tensor as T
import numpy as np

x =T.matrix('input')
target = T.ivector('target')
input_layer = lasagne.layers.InputLayer((None,2))
nnet = lasagne.layers.DenseLayer(input_layer, num_units = 1,nonlinearity = lasagne.nonlinearities.sigmoid)
l_out = lasagne.layers.get_output(nnet,x)
params = lasagne.layers.get_all_params(nnet)
loss = T.mean(T.nnet.binary_crossentropy(l_out,target))
updates_sgd = lasagne.updates.sgd(loss,params, learning_rate = 0.1)
train = theano.function([x,target],updates = updates_sgd)
## Maybe incorrect
gradients = theano.function([x,target],T.grad(loss,params))


X = np.array([[1,2],[1,2],[1,3],[4,2],[4,1],[3,3]]).astype('float32')
Y = np.array([1,1,1,0,0,0]).astype('int32')
## Error in the following call
#gradients (X,Y)
