import theano 
import theano.tensor as T 

from LSTM import LSTMLayer 
from SGD import SGD
from datasets import CharacterDataset


if __name__ == '__main__':

	x = T.tensor3('x')
	y = T.tensor3('y')

	char_ds = CharacterDataset("./data/shakespeare.hdf5")
	char_ds.cut_by_sequence(20)

	lstm = LSTMLayer(x,{'in_dim':char_ds.char_len,'hid_dim':150,'out_dim':char_ds.char_len})
	lstm.load_params("./checkpoints/LSTM_12:41_1.360.hdf5")
	trainer = SGD(lstm, char_ds)

	trainer.compile_functions(x,y,method='rmsprop')

	# trainer.train(0.001,0.9,50)

	trainer.calc_error()