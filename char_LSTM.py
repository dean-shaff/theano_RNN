import time

import theano 
import theano.tensor as T 

from LSTM import LSTMLayer, LSTMMultiLayer
from SGD import SGD
from datasets import CharacterDataset


if __name__ == '__main__':

	cur_time = time.strftime("%d-%m-%y:%H:%M")

	logfile = './logs/LSTM_char_run{}.log'.format(cur_time)

	x = T.tensor3('x')
	y = T.tensor3('y')

	char_ds = CharacterDataset("./data/shakespeare.hdf5")
	char_ds.cut_by_sequence(50)

	# lstm = LSTMLayer(x,{'in_dim':char_ds.char_len,'hid_dim':150,'out_dim':char_ds.char_len},
	# 				logfile=logfile)

	lstm = LSTMMultiLayer(x, 
							[
								{'in_dim':char_ds.char_len,'hid_dim':150,'out_dim':char_ds.char_len},
								{'in_dim':150,'hid_dim':100,'out_dim':char_ds.char_len}
							]
							)
	# lstm.load_params("./checkpoints/LSTM_15:08_1.195.hdf5")

	trainer = SGD(lstm, char_ds,
					logfile=logfile)

	trainer.compile_functions(x,y,method='rmsprop')

	trainer.train(0.0002,0.9,30)

	# trainer.calc_error()