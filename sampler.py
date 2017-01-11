import time 
import pdb
import logging

import numpy as np 
import theano
import theano.tensor as T 


class CharSampler(object):
	"""
	Using the sequence prediction LSTM model, sample from LSTM output distribution.
	"""
	def __init__(self, model, dataset):
		"""
		args:
			- model: An instance of LSTMLayer. Model must have a sequence_sampling method.
			- dataset: An instance of a Dataset. Dataset training sets must be indexable.
		"""
		self.model = model
		self.ds = dataset 

	def compile_functions(self,x,**kwargs):
		"""
		Compile the functions for training the model.
		args:
			- x: The symbolic input to the computational graph
			- y: The symbolic target for the computational graph 
		kwargs: 
			- 
		"""

		n_steps = T.scalar('n_steps', dtype='int64')

		gen_prob = self.model.sequence_sampling(x,n_steps)
		
		t0 = time.time()
		self.gen_prob_fn = theano.function(
			inputs = [x, n_steps],
			outputs = gen_prob
		)

		print("Time compiling probability distribution generator: {:.3f}".format(time.time() - t0))
		return self.gen_prob_fn

	def sample(self, seed, n_steps):
		"""
		Sample from the LSTM, using the function created in compile_functions
		args:
			- seed (str): The letter we want to use to sample
			- n_steps (int): The number of samples to generate			
		"""
		try:
			getattr(self, 'gen_prob_fn')
		except AttributeError:
			logging.error("You need to call compile_functions before calling this function.")
			return 

		# First thing we need to do is get the seed in the correct data format. 
		n_char = len(self.ds.chars)
		char = list(self.ds.chars)
		ind = char.index(seed)
		one_hot = np.zeros(n_char)
		one_hot[ind] = 1.0
		one_hot = one_hot.reshape((1,n_char))
		# Now we can sample from the gen_prob_fn
		prob_distr = self.gen_prob_fn(one_hot, n_steps)
		# prob_distr = prob_distr.reshape((n_steps, n_char))
		# pdb.set_trace()

		prob_distr_max = np.argmax(prob_distr, axis=2)
		prob_distr_sampled = [np.random.choice(n_char,1,p=i[0])[0] for i in prob_distr]

		max_char = [char[i] for i in prob_distr_max.reshape(prob_distr_max.shape[0])]
		sampled_char = [char[i] for i in prob_distr_sampled]

		max_str = "".join(max_char)
		samples_str = "".join(sampled_char)

		print(samples_str)
		print(max_str)




