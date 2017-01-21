import time 
import gzip
import cPickle 
import pdb

import numpy as np 
import theano 
import theano.tensor as T 

from LSTM import LSTMLayer, LSTMMultiLayer
from SGD import SGD
from datasets import MNISTDataset

if __name__ == '__main__':
	x = T.tensor3('x')
	y = T.lvector('y')

	mnist = MNISTDataset("./data/mnist.pkl.gz")

	# # single layer approach
	# hid_dim = 100
	# lstmlayer = LSTMLayer(x,{'in_dim':28,'hid_dim':hid_dim,'out_dim':10})

	# trainer = SGD(lstmlayer, mnist)

	# trainer.compile_functions(x,y,method='rmsprop')

	# trainer.train(0.001,0.9,50)	

	# multlayer approach
	cur_time = time.strftime("%d-%m-%y:%H:%M")

	logfile = './logs/LSTM_MNIST_run{}.log'.format(cur_time)

	hid1 = 100
	hid2 = 50
	lstm = LSTMMultiLayer(x,
						[
							{'in_dim':28,'hid_dim':hid1,'out_dim':10},
							{'in_dim':hid1,'hid_dim':hid2,'out_dim':10}
						],
					logfile=logfile)

	trainer = SGD(lstm, mnist, logfile=logfile)

	trainer.compile_functions(x,y,method='rmsprop')

	trainer.train(0.001,0.9,50)	

