import time 
import pdb
import logging

import numpy as np 
import theano
import theano.tensor as T 

cur_time = time.strftime("%d-%m-%y:%H:%M")

logging.basicConfig(filename='./logs/SGD{}.log'.format(cur_time), format='%(levelname)s:%(message)s', level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler())
# logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

class SGD(object):
	"""
	Implements stochastic gradient descent for training LSTMs 
	"""
	def __init__(self, model, dataset):
		"""
		"""
		self.model = model
		self.ds = dataset 

	def compile_functions(self,x,y,**kwargs):
		"""
		Compile the functions for training the model.
		args:
			- x: The symbolic input to the computational graph
			- y: The symbolic target for the computational graph 
		kwargs:
			- met: The training method to be used ('vanilla', 'adagrad', 'nesterov')
			Based on methods in arXiv:1609.04747
				- Vanilla: The basic SGD without momentum
				- Momentum: SGD with momentum -- this means that it takes into account previous 
					parameter updates so as to avoid annoying local minima in the loss funciton.
				- Neterov: A variant of momentum method where you take gradients with respect not to 
					just the model paramters, but the model parameters minus some fraction of the 
					previous update. 
				- 
		"""
		self.method = kwargs.get('method','rmsprop')

		mb = T.scalar('mb', dtype='int64')
		lr = T.scalar('lr')
		momentum = T.scalar('momentum')
		index = T.scalar('index', dtype='int64')


		if (len(self.ds.test_obs.data.get_value().shape) == 1):
			logging.info("\nPerforming classification task.")
			cost = self.model.log_cost_classify(y)
			error = self.model.error_classify(y)
		else:
			logging.info("\nPerforming sequence classification task.")
			cost = self.model.log_cost_sequence(y)
			error = self.model.error_sequence(y)
		
		grad_params = [] 
		updates = [] 
		if (self.method == 'vanilla'):
			print("Using Vanilla SGD")
			for param in self.model.params:
				grad_p = T.grad(cost, param)
				# grad_clipped = T.clip(grad_p, -1, 1)
				grad_params.append(grad_p)
				updates.append((param, param - lr*momentum*grad_p))

		elif (self.method == 'momentum'):
			print("Using simple momentum scheme")
			for param in self.model.params:
				param_update = theano.shared(param.get_value()*0.)
				grad_cost = T.grad(cost, param)
				# grad_clipped = T.clip(grad_cost, -1, 1)
				updates.append((param_update, momentum*param_update + lr*grad_cost))
				updates.append((param, param - param_update))

		elif (self.method == 'rmsprop'):
			print("Using rmsprop method")
			for param in self.model.params:
				param_update = theano.shared(param.get_value()*0.0)
				grad_cost = T.grad(cost, param)

				updates.append((param_update, momentum*param_update + ((1.0 - momentum) * grad_cost**2)))
				updates.append((param, param - (lr * grad_cost / T.sqrt(param_update + 1e-6))))



		t0 = time.time() 
		logging.info("Compiling theano functions...")

		self.train_fn = theano.function(
			inputs = [index, lr, momentum,mb],
			outputs = cost,
			updates = updates,
			givens = {
				x: self.ds.train_in[(index*mb):(index+1)*mb],
				y: self.ds.train_obs[(index*mb):(index+1)*mb]
			}
		)
		self.error_fn = theano.function(
			inputs = [x,y],
			outputs = error 
		)

		self.raw_output_fn = theano.function(
			inputs = [x],
			outputs = self.model.pred
		)

		logging.info("Time compiling: {:.4f}".format(time.time() - t0))

		return self.train_fn, self.error_fn, self.raw_output_fn

	def calc_error(self):
		"""
		For computing error outside of the training function. Because we have to grab the 
		values of theano shared variables everytime this function is called, I wouldn't 
		use it more than in one off cases. For example, if you want to test the error rate of 
		a checkpoint outside of training, you would use this function.
		"""
		t0 = time.time() 
		test_in = self.ds.test_in.data.get_value()
		test_obs = self.ds.test_obs.data.get_value()
		logging.info("Time loading in shared variables: {:.3f}".format(time.time() - t0))
		t0 = time.time() 
		err = self.error_fn(test_in, test_obs)
		logging.info("Current error: {}".format(err))
		logging.info("Time calculating error: {:.3f}".format(time.time() -t0))
		return err

	def train(self, *args, **kwargs):
		"""
		Train the compiled functions for training.
		Note that calling the error function is very expensive. 
		args:
			- lr: (float) the learning rate to use 
			- momentum: (float) the momentum to use 
			- mb: (int) the minibatch size
		kwargs:
			- test_rate: (int) after how many minibatches to test the model? If -1 does not test
			- save_rate: (int) How frequently to save model parameters? If -1 does automatically.
			- nepochs: (int) The number of epochs for which to run the trainer. (1000)
		"""

		try:
			getattr(self, "train_fn")
		except AttributeError:
			logging.error("You need to call compile_functions before calling this function.")
			return 

		test_rate = kwargs.get('test_rate',-1)
		save_rate = kwargs.get('save_rate',-1)
		nepochs = kwargs.get('nepochs',1000)

		lr, momentum, mb = args
		print(lr, momentum, mb)
		if self.method == 'vanilla':
			momentum = 1.0

		test_amount = 1000

		test_in = self.ds.test_in.data.get_value()[:test_amount]
		test_obs = self.ds.test_obs.data.get_value()[:test_amount]

		train_batches = self.ds.get_train_batches(mb)
		logging.info("\nStarting training. There are {} iterations per epoch.".format(train_batches))
		best_cost = 100. 

		for epoch in xrange(nepochs):
			logging.info("Current epoch {}".format(epoch))
			accum_cost = 0 
			for i in xrange(train_batches):
				t0 = time.time()
				cur_cost = self.train_fn(i, lr, momentum, mb)
				accum_cost += cur_cost
				eval_time = time.time() - t0
				if (best_cost - (accum_cost / float(i)) > 0.005):
					best_cost = accum_cost / float(i)
					logging.info("Current mean cost: {}".format(best_cost))
					logging.info("Current instantaneous cost: {}".format(cur_cost))
					logging.info("Time calculating cost: {:.4f}".format(time.time()- t0))
					cur_time = time.strftime("%H:%M")
					self.model.save_params("./checkpoints/LSTM_{}_{:.3f}.hdf5".format(cur_time, best_cost))


				if (i % test_rate == 0 and test_rate != -1):
					cur_err_test = self.error_fn(test_in, test_obs)
					# cur_err_test = error(test_in[:,:test_amount,:],test_obs[:test_amount])
					print("Current test error: {}\n".format(cur_err_test))
					# cur_err_train = error(train_in[:,:test_amount,:], train_obs[:test_amount])
					# print("Current train error: {}".format(cur_err_train))
