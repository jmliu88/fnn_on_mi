import lasagne
import theano
import theano.tensor as T
import numpy as np

N_FEAT_X = 2
N_FEAT_U = 1
x =T.matrix('input')
u =T.matrix('input2')
target = T.matrix('target')
#target = T.ivector('target')

layers = {'input_x':lasagne.layers.InputLayer((None,N_FEAT_X))}
layers['input_u']=lasagne.layers.InputLayer((None,N_FEAT_U))
layers['v']=lasagne.layers.DenseLayer(layers['input_u'],num_units = 1, nonlinearity=lasagne.nonlinearities.linear, W = lasagne.init.Constant(1.), b= lasagne.init.Constant(0.))
layers['a']=lasagne.layers.NonlinearityLayer(layers['v'], nonlinearity=lasagne.nonlinearities.sigmoid)
weights = T.tile(lasagne.layers.get_output(layers['a'], u ), (1, N_FEAT_X))
x_hat = weights * x
compute_weights= theano.function([u],weights)
weighted_input = theano.function([x,u],x_hat)

params = lasagne.layers.get_all_params(layers['a'])
loss = T.mean((x_hat.flatten()-target.flatten())**2)
#loss = T.mean(T.nnet.binary_crossentropy(x_hat,target))
#updates_sgd = lasagne.updates.sgd(loss,params, learning_rate = 0.1)
#train = theano.function([x,target],updates = updates_sgd)
gradients = theano.function([x,u,target],T.grad(loss,params))


X = np.array([[1,2],[1,2],[1,3],[4,2],[4,1],[3,3]]).astype('float32')
U = np.array([[0],[1],[1],[2],[3],[99]]).astype('float32')
print X
print compute_weights(U)
print weighted_input(X,U)
#Y = np.array([1,1,1,0,0,0]).astype('int32')
print gradients (X,U,X)
