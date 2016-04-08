import lasagne
import theano
import theano.tensor as T
import numpy as np

N_FEAT_X = 2
N_FEAT_U = 1
x =T.matrix('input')
u =T.matrix('input2')
target = T.ivector('target')

layers = {'input_x':lasagne.layers.InputLayer((None,N_FEAT_X))}
layers['input_u']=lasagne.layers.InputLayer((None,N_FEAT_U))
layers['v']=lasagne.layers.DenseLayer(layers['input_u'],num_units = 1, nonlinearity=lasagne.nonlinearities.linear, W = lasagne.init.Constant(1.), b= lasagne.init.Constant(0.))
layers['a']=lasagne.layers.NonlinearityLayer(layers['v'], nonlinearity=lasagne.nonlinearities.sigmoid)
#layers['a']=lasagne.layers.DenseLayer(layers['input_u'],num_units = 1, nonlinearity=lasagne.nonlinearities.linear, W = lasagne.init.Constant(1.), b= lasagne.init.Constant(0.))
weights = T.tile(lasagne.layers.get_output(layers['a'], u ), (1, N_FEAT_X))
x_hat = weights * x
weighted_input = theano.function([x,u],x_hat)
compute_weights= theano.function([u],weights)
#layers['output'] = lasagne.layers.
#input_layer = lasagne.layers.InputLayer((None,N_FEAT_X))
#nnet = lasagne.layers.DenseLayer(input_layer, num_units = 1,nonlinearity = lasagne.nonlinearities.sigmoid)
#l_out = T.flatten(lasagne.layers.get_output(nnet,x))
#params = lasagne.layers.get_all_params(nnet)
#loss = T.mean(T.nnet.binary_crossentropy(l_out,target))
#updates_sgd = lasagne.updates.sgd(loss,params, learning_rate = 0.1)
#train = theano.function([x,target],updates = updates_sgd)
#gradients = theano.function([x,target],T.grad(loss,params))


X = np.array([[1,2],[1,2],[1,3],[4,2],[4,1],[3,3]]).astype('float32')
U = np.array([[0],[1],[1],[2],[3],[99]]).astype('float32')
print X
print compute_weights(U)
print weighted_input(X,U)
#Y = np.array([1,1,1,0,0,0]).astype('int32')
#gradients (X,Y)
