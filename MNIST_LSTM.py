import time 
import gzip
import cPickle 
import pdb

import numpy as np 
import theano 
import theano.tensor as T 

from LSTM import LSTMLayer 
from SGD import SGD
from datasets import MNISTDataset

def MNIST_processor(mnist_file):
    print("Loading data into memory")
    t0 = time.time()
    f = gzip.open(mnist_file, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    print("Time loading data into memory: {:.4f}".format(time.time() - t0))
    def generate_sequences(dset):
    	data_in = dset[0]
    	data_obs = theano.shared(dset[1])
    	data_in = data_in.reshape((data_in.shape[0],28,28))
    	data_in = theano.shared(data_in.swapaxes(0,1).astype(theano.config.floatX))
    	return [data_in, data_obs]


    data_pack = {
    			'train':generate_sequences(train_set),
    			'valid':generate_sequences(valid_set),
    			'test':generate_sequences(test_set)
    			}
    return data_pack

def create_train_comp_graph(x,y):
	"""
	Let's start by calling each image in MNIST a sequence of 28 length vectors. 
	I think if this doesn't work we can try other approaches. 
	Implement the SGD functions here as well.
	"""
	data_pack = MNIST_processor('./data/mnist.pkl.gz')
	# print(data_pack['train'][0].)
	train_in = data_pack['train'][0]
	train_obs = data_pack['train'][1]

	mb = T.scalar('mb', dtype='int64')
	lr = T.scalar('lr')
	index = T.scalar('index', dtype='int64')

	hid_dim = 100
	lstmlayer = LSTMLayer(x,{'in_dim':28,'hid_dim':hid_dim,'out_dim':10})

	prob_final = lstmlayer.pred[-1]

	pred = T.argmax(prob_final,axis=1)

	cost = -T.mean(T.log(prob_final)[T.arange(y.shape[0]),y])
	# cost = T.nnet.categorical_crossentropy(pred, y)

	error = T.mean(T.neq(pred, y))

	grad_params = [T.grad(cost, p) for p in lstmlayer.params]

	updates = [(p, p-lr*g) for p, g in zip(lstmlayer.params, grad_params)]

	t0 = time.time()
	train_model = theano.function(
		inputs = [index, lr, mb],
		outputs = cost, 
		updates = updates, 
		givens = {
			x: train_in[:,(index*mb):(index+1)*mb,:],
			y: train_obs[(index*mb):(index+1)*mb]
		}
	)
	print("Time compiling training function: {:.4f}".format(time.time() - t0))
	t0 = time.time()
	error = theano.function(
		inputs = [x,y],
		outputs = error
	)

	pred_fn = theano.function(
		inputs = [x],
		outputs = prob_final
	)
	print("Time compiling error function: {:.4f}".format(time.time() - t0))

	return train_model, error, pred_fn, data_pack

def train(train_model, error,pred_fn, data_pack, lr, mb, nepochs, **kwargs):

	test_rate = kwargs.get('test_rate', 10)
	test_amount = 200
	print("Starting to train the model...")

	train_batches = int(data_pack['train'][1].get_value().shape[0] // mb)
	test_in, test_obs = data_pack['test'][0].get_value(), data_pack['test'][1].get_value()
	train_in, train_obs = data_pack['train'][0].get_value(), data_pack['train'][1].get_value()

	for epoch in xrange(nepochs):
		accum_cost = 0
		for i in xrange(train_batches):
			t0 = time.time()
			cur_cost = train_model(i, lr, mb)
			eval_time = time.time() - t0
			accum_cost += cur_cost
			if (i % test_rate == 0):
				print("Last evaluation time: {:.4f}".format(eval_time))
				print("Current cost: {}".format(accum_cost / float(test_rate)))

				accum_cost = 0

				cur_err_test = error(test_in[:,:test_amount,:],test_obs[:test_amount])
				print("Current test error: {}".format(cur_err_test))
				cur_err_train = error(train_in[:,:test_amount,:], train_obs[:test_amount])
				print("Current train error: {}".format(cur_err_train))



if __name__ == '__main__':
	x = T.tensor3('x')
	y = T.lvector('y')

	mnist = MNISTDataset("./data/mnist.pkl.gz")

	hid_dim = 100
	lstmlayer = LSTMLayer(x,{'in_dim':28,'hid_dim':hid_dim,'out_dim':10})

	trainer = SGD(lstmlayer, mnist)

	trainer.compile_functions(x,y)

	trainer.train(0.01, 10,1000)	

	# data_pack = MNIST_processor("./data/mnist.pkl.gz")
	# train_model, error, pred_fn, data_pack = create_train_comp_graph(x,y)
	# train(train_model, error, pred_fn, data_pack, 0.001, 10, 1000, test_rate=100)