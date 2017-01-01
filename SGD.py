import time 
import pdb
import logging

import numpy as np 
import theano
import theano.tensor as T 

cur_time = time.strftime("%d-%m-%y:%H:%M")

# logging.basicConfig(filename='./logs/SGD{}.log'.format(cur_time), format='%(levelname)s:%(message)s', level=logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

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
			(I think I'll put something in here about changing up the cost)
		"""

		mb = T.scalar('mb', dtype='int64')
		lr = T.scalar('lr')
		momentum = T.scalar('momentum')
		index = T.scalar('index', dtype='int64')

		cost = self.model.log_cost_classify(y)

		error = self.model.class_error(y)

		grad_params = [] 
		updates = [] 
		for param in self.model.params:
			grad_p = T.grad(cost, param)
			grad_clipped = T.clip(grad_p, -1, 1)
			grad_params.append(grad_clipped)
			updates.append((param, param - lr*grad_clipped))

		t0 = time.time() 
		logging.info("Compiling theano functions...")

		self.train_fn = theano.function(
			inputs = [index, lr, mb],
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

	def train(self, lr, mb, nepochs, **kwargs):
		"""
		Train the compiled functions for training.
		Note that calling the error function is very expensive. 
		args:
			- lr: (float) the learning rate to use 
			- mb: (int) the minibatch size
			- nepochs: (int) The number of epochs for which to run the trainer.
		kwargs:
			- test_rate: (int) after how many minibatches to test the model?
			- save_rate: (int) How frequently to save model parameters? If -1 does automatically.
		"""

		try:
			getattr(self, "train_fn")
		except AttributeError:
			logging.error("You need to call compile_functions before calling this function.")
			return 

		test_rate = kwargs.get('test_rate',50)
		save_rate = kwargs.get('save_rate',-1)

		train_batches = self.ds.get_train_batches(mb)
		logging.info("Starting training...")
		for epoch in xrange(nepochs):
			accum_cost = 0 
			logging.info("Current epoch {}".format(epoch))
			for i in xrange(train_batches):
				t0 = time.time()
				cur_cost = self.train_fn(i, lr, mb)
				eval_time = time.time() - t0
				accum_cost += cur_cost
				if (i % test_rate == 0):
					logging.info("Current cost: {}\n".format(accum_cost / float(test_rate)))
					accum_cost = 0 
					# cur_err_test = error(test_in[:,:test_amount,:],test_obs[:test_amount])
					# print("Current test error: {}".format(cur_err_test))
					# cur_err_train = error(train_in[:,:test_amount,:], train_obs[:test_amount])
					# print("Current train error: {}".format(cur_err_train))
