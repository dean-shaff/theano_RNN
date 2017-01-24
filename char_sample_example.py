import theano 
import theano.tensor as T 

from LSTM import LSTMLayer, LSTMMultiLayer 
from sampler import CharSampler
from datasets import CharacterDataset


if __name__ == '__main__':

	# x = T.tensor3('x')
	x = T.row('x')

	char_ds = CharacterDataset("./data/shakespeare.hdf5")
	char_ds.cut_by_sequence(50)

	#lstm = LSTMLayer(x,)
	lstm = LSTMMultiLayer(x,
								[
									{'in_dim':char_ds.char_len,'hid_dim':150,'out_dim':char_ds.char_len},
									{'in_dim':150,'hid_dim':100,'out_dim':char_ds.char_len}
								])
								
	

	lstm.load_params("./checkpoints/LSTM_21:10_1.215.hdf5")

	sampler = CharSampler(lstm, char_ds)
	sampler.compile_functions(x)
	sampler.sample('d',50)
