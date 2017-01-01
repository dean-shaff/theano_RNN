import time 

import numpy as np 
import theano 
import theano.tensor as T 

from LSTM import LSTMLayer 
from SGD import SGD
from datasets import CharacterDataset


if __name__ == '__main__':

	x = T.tensor3('x')
	y = T.tensor3('y')

	char_ds = CharacterDataset("./data/shakespeare.hdf5")
	char_ds.cut_by_sequence(50)

	lstm = LSTMLayer(x,{'in_dim':char_ds.char_len,'hid_dim':150,'out_dim':char_ds.char_len})

	trainer = SGD(lstm, char_ds)

	trainer.compile_functions(x,y)

	trainer.train(0.001,10,1000)