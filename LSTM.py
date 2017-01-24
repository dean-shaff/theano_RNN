import time
import pdb

import numpy as np 
import theano 
import theano.tensor as T 
import h5py

from util import logging_config 

class LSTMMultiLayer(object):

	def __init__(self,X,dims,**kwargs):
		"""
		Create the computational graph for a multilayer LSTM.
		args:
			- X (theano.tensor.tensor): The symblic input to the computational graph
			- dim (list): A list of dictionaries containing the dimensions of each unit in the LSTM. 
				This gets passed to LSTMLayer
		kwargs:
			- logfile (str): The logfile to be used for file logging (None).
			- bptt_truncate (int): The number of steps after which to stop propagating gradients (-1).
		"""
		logfile = kwargs.get('logfile',None)
		truncate = kwargs.get("bptt_truncate", -1)

		self.logger = logging_config("LSTM_multilayer",logfile=logfile)

		uni = np.random.uniform
		
		self.dims = dims
		self.layers = []
		self.params = []

		for d in dims:
			li = LSTMLayer(X,d,logfile=logfile,weights_only=True)
			self.layers.append(li)
			self.params += li.params

		nhid_final = dims[-1].get('hid_dim')
		nout_final = dims[-1].get('out_dim')
		self.Wy = theano.shared(uni(-np.sqrt(1.0/(nhid_final*nout_final)), np.sqrt(1.0/(nhid_final*nout_final)),(nhid_final,nout_final)).astype(theano.config.floatX),name='Wy')
		self.by = theano.shared(np.zeros(nout_final), name='by')
		self.params += [self.Wy, self.by]

		def recurrent_step(x_t,*args): #b_tm1,s_tm1):
			"""
			Define the recurrent step for the multilayer network.
			args:
				- x_t: the current sequence
				- b_tm1: the previous b_t (b_{t minus 1})
				- s_tml: the previous s_t (s_{t minus 1}) this is the state of the cell
			"""
			b_tm1, s_tm1 = args[0], args[1]
			outputs = []

			l0 = self.layers[0]
			# Input 
			b_L = T.nnet.sigmoid(T.dot(x_t, l0.Wi[0]) + T.dot(b_tm1,l0.Wh[0]) + T.dot(s_tm1, l0.Wc[0]) + l0.b[0])
			# Forget
			b_Phi = T.nnet.sigmoid(T.dot(x_t,l0.Wi[1]) + T.dot(b_tm1,l0.Wh[1]) + T.dot(s_tm1, l0.Wc[1]) + l0.b[1])
			# Cell 
			a_Cell = T.dot(x_t, l0.Wi[2]) + T.dot(b_tm1, l0.Wh[2]) + l0.b[2]
			s_t = b_Phi * s_tm1 + b_L*T.tanh(a_Cell)
			# Output 
			b_Om = T.nnet.sigmoid(T.dot(x_t, l0.Wi[3]) + T.dot(b_tm1,l0.Wh[3]) + T.dot(s_t, l0.Wc[2]) + l0.b[3])
			# Final output (What gets sent to the next step in the recurrence) 
			b_Cell = b_Om*T.tanh(s_t)

			outputs.append(b_Cell)
			outputs.append(s_t)

			for i in xrange(1, len(self.layers)):
				li = self.layers[i]
				b_tm1, s_tm1 = args[2*i], args[(2*i)+1]
				# Input 
				b_L = T.nnet.sigmoid(T.dot(b_Cell, li.Wi[0]) + T.dot(b_tm1,li.Wh[0]) + T.dot(s_tm1, li.Wc[0]) + li.b[0])
				# Forget
				b_Phi = T.nnet.sigmoid(T.dot(b_Cell,li.Wi[1]) + T.dot(b_tm1,li.Wh[1]) + T.dot(s_tm1, li.Wc[1]) + li.b[1])
				# Cell 
				a_Cell = T.dot(b_Cell, li.Wi[2]) + T.dot(b_tm1, li.Wh[2]) + li.b[2]
				s_t = b_Phi * s_tm1 + b_L*T.tanh(a_Cell)
				# Output 
				b_Om = T.nnet.sigmoid(T.dot(b_Cell, li.Wi[3]) + T.dot(b_tm1,li.Wh[3]) + T.dot(s_t, li.Wc[2]) + li.b[3])
				# Final output (What gets sent to the next step in the recurrence) 
				b_Cell = b_Om*T.tanh(s_t)
				outputs.append(b_Cell)
				outputs.append(s_t)

			# Sequence output
			o_t = T.nnet.softmax(T.dot(outputs[-2], self.Wy) + self.by)
			
			return_vals = [o_t] + outputs

			return return_vals

		self.recurrent_step = recurrent_step

		outputs_info = [{'initial':None}]
		for i in xrange(len(dims)):
			outputs_info.append({'initial':T.zeros((X.shape[1],dims[i]['hid_dim']))})
			outputs_info.append({'initial':T.zeros((X.shape[1],dims[i]['hid_dim']))})


		out, _ = theano.scan(self.recurrent_step,
									truncate_gradient=truncate,
									sequences=X,
									outputs_info=outputs_info,
									n_steps=X.shape[0])

		# self.b_out = out[1]
		self.pred = out[0]
	
	def sequence_sampling(self,seed,n_steps):
		"""
		Define the outputs for LSTM that feeds back into itself. Basically the same thing as in __init__
		but we feed output back in as input.
		"""
		outputs_info = [{'initial':seed}]
		for i in xrange(len(self.dims)):
			outputs_info.append({'initial':T.zeros((1,self.dims[i]['hid_dim']))}) 		
			outputs_info.append({'initial':T.zeros((1,self.dims[i]['hid_dim']))}) 		
	
		out, _ = theano.scan(self.recurrent_step,
								outputs_info=outputs_info
								n_steps=n_steps)

		seq_pred = out[0]
		return seq_pred



	def save_params(self,filename,**kwargs):
		"""
		Save current model parameters to a specified filename.
		args:
			- filename (str): The name of the hdf5 file in which to save the model parameters.
		kwargs:
			- any training parameters we want to save. 
		"""
		self.logger.info("Saving model parameters.")
		t0 = time.time()
		f = h5py.File(filename,'w')
		for i,layer in enumerate(self.layers):
			grp = f.create_group('l{}'.format(i))
			for param in layer.params:
				grp.create_dataset(param.name, data=param.get_value())
		f.create_dataset('Wy', data=self.Wy.get_value())
		f.create_dataset('by', data=self.by.get_value())
		self.logger.info("Saving complete. Time spent: {:.3f}\n".format(time.time() - t0))
		f.close()

	def load_params(self, filename):
		"""
		Load in some model parameters from a file. 
		args:
			- filename (str): The name of the hdf5 file containing the model parameters.
		"""
		self.logger.info("Loading in model parameters")
		t0 = time.time() 
		f = h5py.File(filename,'r')
		for i, layer in enumerate(self.layers):
			grp = f['/l{}'.format(i)]
			for param in layer.params:
				param_val = grp[param.name][...]
				param.set_value(param_val)
		self.Wy.set_value(f['Wy'][...])
		self.by.set_value(f['by'][...])
		self.logger.info("Loading complete. Time spent: {:.3f}\n".format(time.time() - t0))

	def neg_log_likelihood(self, x,y):

		# y_arg_max = T.argmax(y, axis=1)
		return -T.mean(T.log(x)[T.arange(y.shape[0]),y])

	def log_cost_sequence(self, y):

		log_like, _ = theano.scan(self.neg_log_likelihood, 
								sequences=[self.pred,T.argmax(y,axis=2)])

		return T.mean(log_like)

	def error(self, x,y):
		"""
		The generic function -- calculates error between two 2d tensors. 
		"""
		return T.mean(T.neq(T.argmax(x, axis=1),T.argmax(y, axis=1)))

	def error_sequence(self,y):

		# err, _ = theano.scan(self.error, 
		#                     sequences = [self.pred,y])        

		err = T.mean(T.neq(T.argmax(self.pred, axis=2), T.argmax(y, axis=2)))

		return err


	def log_cost_classify(self,y):

		return self.neg_log_likelihood(self.pred[-1],y)

	def error_classify(self,y):

		return T.mean(T.neq(T.argmax(self.pred[-1],axis=1), y)) 


class LSTMLayer(object):

	def __init__(self,X,dim,**kwargs):
		"""
		Set up the weight matrices for a long short term memory (LSTM) unit. 
		I use the notation from Graves. 
		args:
			- X (theano.tensor.tensor): The symblic input to the computational graph
			- dim: A dictionary containing the dimensions of the units inside the LSTM.  
		kwargs:
			- logfile (str): the logfile to be used for logging (None)
			- weights_only (bool): Whether or not to just setup weights.
		"""
		logfile = kwargs.get('logfile',None)
		weights_only = kwargs.get('weights_only',False)

		self.logger = logging_config('LSTM_layer',logfile=logfile)

		uni = np.random.uniform

		def diag_constructor(limit,size,n):
			"""
			args:
				- limit: A list whose two elements correspond to the limit for the numpy uniform function.
				- size: (Int) one dimension of the square matrix.
				- n: The number of these matrices to create.
			"""

			diag_ind = np.diag_indices(size)
			mat = np.zeros((n,size,size))
			for i in xrange(n):
				diag_val = uni(limit[0], limit[1],size)
				mat[i,diag_ind] = diag_val
			return mat.astype(theano.config.floatX)          


		truncate = kwargs.get("bptt_truncate", -1)

		nin = dim.get('in_dim')
		nout = dim.get('out_dim')
		nhid = dim.get('hid_dim')
		self.nin = nin
		self.nout = nout 
		self.nhid = nhid 
		# I can cast weight matrices differently. Instead of creating separate weight matrices for each connection, I create them 
		# based on their size. This cleans up the code and potentially makes things more efficient. I will say that it makes 
		# the recurrent step function harder to read.
		self.Wi = theano.shared(uni(-np.sqrt(1.0/(nin*nhid)), np.sqrt(1.0/(nin*nhid)),(4, nin, nhid)).astype(theano.config.floatX),name='Wi')
		self.Wh = theano.shared(uni(-np.sqrt(1.0/(nhid**2)), np.sqrt(1.0/(nhid**2)),(4, nhid, nhid)).astype(theano.config.floatX),name='Wh')
		self.Wc = theano.shared(diag_constructor([-np.sqrt(1.0/(nhid**2)), np.sqrt(1.0/(nhid**2))],nhid,3),name='Wc')
		self.b = theano.shared(np.zeros((4,nhid)), name='b')
		
		if weights_only:

			self.params = [self.Wi, self.Wh, self.Wc, self.b]

		if not weights_only:

			self.Wy = theano.shared(uni(-np.sqrt(1.0/(nhid*nout)), np.sqrt(1.0/(nhid*nout)),(nhid,nout)).astype(theano.config.floatX),name='Wy')
			self.by = theano.shared(np.zeros(nout), name='by')

			self.params = [self.Wi, self.Wh, self.Wc, self.b, self.Wy, self.by]


			def recurrent_step(x_t,b_tm1,s_tm1):
				"""
				Define the recurrent step.
				args:
					- x_t: the current sequence
					- b_tm1: the previous b_t (b_{t minus 1})
					- s_tml: the previous s_t (s_{t minus 1}) this is the state of the cell
				"""
				# Input 
				b_L = T.nnet.sigmoid(T.dot(x_t, self.Wi[0]) + T.dot(b_tm1,self.Wh[0]) + T.dot(s_tm1, self.Wc[0]) + self.b[0])
				# Forget
				b_Phi = T.nnet.sigmoid(T.dot(x_t,self.Wi[1]) + T.dot(b_tm1,self.Wh[1]) + T.dot(s_tm1, self.Wc[1]) + self.b[1])
				# Cell 
				a_Cell = T.dot(x_t, self.Wi[2]) + T.dot(b_tm1, self.Wh[2]) + self.b[2]
				s_t = b_Phi * s_tm1 + b_L*T.tanh(a_Cell)
				# Output 
				b_Om = T.nnet.sigmoid(T.dot(x_t, self.Wi[3]) + T.dot(b_tm1,self.Wh[3]) + T.dot(s_t, self.Wc[2]) + self.b[3])
				# Final output (What gets sent to the next step in the recurrence) 
				b_Cell = b_Om*T.tanh(s_t)
				# Sequence output
				o_t = T.nnet.softmax(T.dot(b_Cell, self.Wy) + self.by)

				return o_t, b_Cell, s_t 
			

			self.recurrent_step = recurrent_step

			out, _ = theano.scan(self.recurrent_step,
									truncate_gradient=truncate,
									sequences = X,
									outputs_info=[
													{'initial':None},
													{'initial':T.zeros((X.shape[1],nhid))},
													{'initial':T.zeros((X.shape[1],nhid))}
												],
									n_steps=X.shape[0])



			self.b_out = out[1]
			self.pred = out[0]

	def sequence_sampling(self,seed,n_steps):
		"""
		Define the outputs for LSTM that feeds back into itself. Basically the same thing as in __init__
		but we feed output back in as input.
		"""

		out, _ = theano.scan(self.recurrent_step,
								outputs_info=[
												{'initial':seed},                                                
												{'initial':T.zeros((1,self.nhid))},
												{'initial':T.zeros((1,self.nhid))}
											],
								n_steps=n_steps)

		seq_pred = out[0]
		return seq_pred


	def save_params(self,filename,**kwargs):
		"""
		Save current model parameters to a specified filename.
		args:
			- filename (str): The name of the hdf5 file in which to save the model parameters.
		kwargs:
			- any training parameters we want to save. 
		"""
		self.logger.info("Saving model parameters.")
		t0 = time.time()
		f = h5py.File(filename,'w')
		for param in self.params:
			f.create_dataset(param.name, data=param.get_value())
		self.logger.info("Saving complete. Time spent: {:.3f}\n".format(time.time() - t0))
		f.close()

	def load_params(self, filename):
		"""
		Load in some model parameters from a file. 
		args:
			- filename (str): The name of the hdf5 file containing the model parameters.
		"""
		self.logger.info("Loading in model parameters")
		t0 = time.time() 
		f = h5py.File(filename,'r')
		for param in self.params:
			param_val = f[param.name][...]
			param.set_value(param_val)
		self.logger.info("Loading complete. Time spent: {:.3f}\n".format(time.time() - t0))


	def neg_log_likelihood(self, x,y):

		# y_arg_max = T.argmax(y, axis=1)
		return -T.mean(T.log(x)[T.arange(y.shape[0]),y])

	def log_cost_sequence(self, y):

		log_like, _ = theano.scan(self.neg_log_likelihood, 
								sequences=[self.pred,T.argmax(y,axis=2)])

		return T.mean(log_like)

	def error(self, x,y):
		"""
		The generic function -- calculates error between two 2d tensors. 
		"""
		return T.mean(T.neq(T.argmax(x, axis=1),T.argmax(y, axis=1)))

	def error_sequence(self,y):

		# err, _ = theano.scan(self.error, 
		#                     sequences = [self.pred,y])        

		err = T.mean(T.neq(T.argmax(self.pred, axis=2), T.argmax(y, axis=2)))

		return err


	def log_cost_classify(self,y):

		return self.neg_log_likelihood(self.pred[-1],y)

	def error_classify(self,y):

		return T.mean(T.neq(T.argmax(self.pred[-1],axis=1), y))



if __name__ == "__main__":
	x = T.tensor3('x')

	lstm = LSTMMultiLayer(x,[
							{'in_dim':50,'hid_dim':100,'out_dim':20},
							{'in_dim':100,'hid_dim':100,'out_dim':20}])
	lstm.save_params('./checkpoints/testsave_multilayer.hdf5')
	lstm.load_params('./checkpoints/testsave_multilayer.hdf5')
	t0 = time.time()
	f = theano.function([x], lstm.pred)
	print("Took {:.4f} seconds to compile".format(time.time() - t0))
	x0 = np.random.randn(150,75,50) #sequence length,  batch size, character length (input length)
	for i in xrange(10):
		t0 = time.time()
		res = f(x0)
		print("{:.4f} in calculation".format(time.time() - t0))

	# build a basic layer
	# lstm = LSTMLayer(x,{'in_dim':100,'hid_dim':100,'out_dim':20})
	# lstm.save_params("testsave.hdf5")
	# lstm.load_params("testsave.hdf5")
	# t0 = time.time()
	# f = theano.function([x],lstm.pred)
	# print("Took {:.4f} seconds to compile".format(time.time() - t0))
	# x0 = np.random.randn(150,50,100) #sequence length,  batch size, character length (input length)
	# for i in xrange(10):
	# 	t0 = time.time()
	# 	res = f(x0)
	# 	print("{:.4f} in calculation".format(time.time() - t0))

	# pdb.set_trace()
