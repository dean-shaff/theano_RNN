import time 
import pdb
import logging

import numpy as np 
import theano
import theano.tensor as T 

from util import logging_config 

class SGD(object):
	"""
	Implements stochastic gradient descent for training LSTMs 
	"""
	def __init__(self, model, dataset, **kwargs):
		"""
		args:
			- model: An instance of LSTMLayer
			- dataset: An instance of a Dataset 
		kwargs:
			- logfile (str): The name of a logfile
		"""
		logfile = kwargs.get('logfile',None)
		self.SGDlogger = logging_config('Trainer',logfile=logfile)

		# if not logfile:
		# 	cur_time = time.strftime("%d-%m-%y:%H:%M")
		# 	self.SGDlogger = logging_config('SGD',logfile='./logs/SGD{}.log'.format(cur_time))
		# else:
		# 	self.SGDlogger = logging_config('SGD',logfile=logfile)

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
			self.SGDlogger.info("Performing classification task.")
			cost = self.model.log_cost_classify(y)
			error = self.model.error_classify(y)
		else:
			self.SGDlogger.info("Performing sequence classification task.")
			cost = self.model.log_cost_sequence(y)
			error = self.model.error_sequence(y)
		
		grad_params = [] 
		updates = [] 
		if (self.method == 'vanilla'):
			self.SGDlogger.info("Using Vanilla SGD")
			for param in self.model.params:
				grad_p = T.grad(cost, param)
				# grad_clipped = T.clip(grad_p, -1, 1)
				grad_params.append(grad_p)
				updates.append((param, param - lr*momentum*grad_p))

		elif (self.method == 'momentum'):
			self.SGDlogger.info("Using simple momentum scheme")
			for param in self.model.params:
				param_update = theano.shared(param.get_value()*0.)
				grad_cost = T.grad(cost, param)
				# grad_clipped = T.clip(grad_cost, -1, 1)
				updates.append((param_update, momentum*param_update + lr*grad_cost))
				updates.append((param, param - param_update))

		elif (self.method == 'rmsprop'):
			self.SGDlogger.info("Using rmsprop method")
			for param in self.model.params:
				param_update = theano.shared(param.get_value()*0.0)
				grad_cost = T.grad(cost, param)

				updates.append((param_update, momentum*param_update + ((1.0 - momentum) * grad_cost**2)))
				updates.append((param, param - (lr * grad_cost / T.sqrt(param_update + 1e-6))))



		t0 = time.time() 
		self.SGDlogger.info("Compiling theano functions...")

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

		self.SGDlogger.info("Time compiling: {:.4f}".format(time.time() - t0))

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
		self.SGDlogger.info("Time loading in shared variables: {:.3f}".format(time.time() - t0))
		t0 = time.time() 
		err = self.error_fn(test_in, test_obs)
		self.SGDlogger.info("Current error: {}".format(err))
		self.SGDlogger.info("Time calculating error: {:.3f}".format(time.time() -t0))
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
			self.SGDlogger.error("You need to call compile_functions before calling this function.")
			return 

		test_rate = kwargs.get('test_rate',-1)
		save_rate = kwargs.get('save_rate',-1)
		nepochs = kwargs.get('nepochs',1000)

		lr, momentum, mb = args

		if self.method == 'vanilla':
			momentum = 1.0

		test_amount = 100

		reset_rate = -1

		test_in = self.ds.test_in.data.get_value()[:test_amount]
		test_obs = self.ds.test_obs.data.get_value()[:test_amount]

		train_batches = self.ds.get_train_batches(mb)
		self.SGDlogger.info("\nStarting training. There are {} iterations per epoch.".format(train_batches))
		best_cost = 100. 

		for epoch in xrange(nepochs):
			# cur_err = self.error_fn(test_in, test_obs)
			self.SGDlogger.info("\nCurrent epoch: {}".format(epoch))
			# self.SGDlogger.info("Current error: {}".format(cur_err))
			accum_cost = 0.
			accum_count = 0.
			for i in xrange(1,train_batches):
				t0 = time.time()
				cur_cost = self.train_fn(i, lr, momentum, mb)
				accum_cost += cur_cost
				accum_count += 1.
				eval_time = time.time() - t0
				if (i % 100 == 0):
					self.SGDlogger.info("Iteration: {}. Current instantaneous cost: {}".format(i, cur_cost))

				# test to see if we're making improvement.
				if (best_cost - accum_cost/accum_count > 0.005):
					best_cost = accum_cost/accum_count

					self.SGDlogger.info("Iteration: {}. Current mean cost: {}".format(i,best_cost))
					# self.SGDlogger.info("Current instantaneous cost: {}".format(cur_cost))
					self.SGDlogger.info("Time calculating cost: {:.4f}".format(time.time()- t0))
					cur_time = time.strftime("%H:%M")
					self.model.save_params("./checkpoints/LSTM_{}_{:.3f}.hdf5".format(cur_time, float(best_cost)))

				#reset the accumulated cost
				if (i % reset_rate == 0 and reset_rate != -1):
					accum_cost = cur_cost
					accum_count = 0.
