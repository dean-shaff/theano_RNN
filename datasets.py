import time 
import pdb
import cPickle
import gzip
import logging 

import numpy as np 
import theano 
import theano.tensor as T 
import h5py

from util import logging_config 

class Datapack(object):
    """
    A class that represents individual data units 
    eg training input data, training observation data. 
    """
    def __init__(self, data,dtype):
        self.data = data 
        self.dtype = dtype

    def __getitem__(self,ind):
        """
        Not super happy with this solution right now.
        """
        if self.dtype == 'sequence':
            return self.data[:,ind,:]
        elif self.dtype == 'vector':
            return self.data[ind]

class CharacterDataset(object):

    def __init__(self, filename,**kwargs):

        logfile = kwargs.get('logfile',None)
        self.DSlogger = logging_config('Char_DS',logfile=logfile)

        self.filename = filename
        self.DSlogger.info("Loading data into memory")
        t0 = time.time() 
        f = h5py.File(self.filename,'r') 
        dat = f['dat'][...]
        dat = dat[:int(dat.shape[0]*0.5)]
        self.DSlogger.info("Data loaded from hdf5 file. Took {:.2f} seconds\n".format(time.time() - t0))
        chars = f.attrs['chars'] 
        dat_one_hot = np.zeros((dat.shape[0], len(chars)),dtype=theano.config.floatX) #need float because we're doing regression 
        dat_one_hot[np.arange(dat.shape[0]),dat] = 1.0 
        self.DSlogger.info("One hot vectors generated. Took {:.2f} second\n".format(time.time() - t0))
        input_dat = dat_one_hot[:-1]
        obs_dat = dat_one_hot[1:]
        f.close() 

        self.chars = chars
        self.char_len = len(chars) 
        self.data = dat 
        self.data_one_hot = dat_one_hot 

        self.obs_dat = obs_dat 
        self.input_dat = input_dat

    def cut_by_sequence(self, seq_len):
        """
        Cut up the dataset into matrices, each (seq_len x char_len) big.
        This also creates training and testing datasets. 
        args:
            - seq_len: The length of the sequence 
        """
        self.seq_len = seq_len
        self.DSlogger.info('Creating shared variables...') 
        t0 = time.time()
        n_seq = (self.data.shape[0] // seq_len) -1

        obs_dat = self.obs_dat[:n_seq*seq_len,:].reshape((n_seq,seq_len,self.char_len))
        self.train_obs = Datapack(theano.shared(obs_dat[:int(0.6*n_seq),:,:].swapaxes(0,1)),"sequence")
        self.test_obs = Datapack(theano.shared(obs_dat[int(0.6*n_seq):,:,:].swapaxes(0,1)),"sequence")

        in_dat = self.input_dat[:n_seq*seq_len,:].reshape((n_seq,seq_len,self.char_len))
        self.train_in = Datapack(theano.shared(in_dat[:int(0.6*n_seq),:,:].swapaxes(0,1)),"sequence")
        self.test_in = Datapack(theano.shared(in_dat[int(0.6*n_seq):,:,:].swapaxes(0,1)),"sequence")
        self.DSlogger.info("Created shared variables. Took {:.2f} seconds.\n".format(time.time() - t0))
        

    def get_train_batches(self, mbsize):
        """
        args:
            - mbsize (int): The size of the minibatch
        """
        return int(self.train_obs.data.get_value().shape[1] // mbsize)

class MNISTDataset(object):
    """
    A class that contains the training, testing and validation sets for MNIST.
    """
    def __init__(self, filename,**kwargs):
        """
        args:
            - filename (str): The filename of the MNIST pickle file.
        """    
        logfile = kwargs.get('logfile',None)
        self.DSlogger = logging_config('MNIST_DS',logfile=logfile)

        self.DSlogger.info("Loading data into memory")
        t0 = time.time()
        f = gzip.open(filename, 'rb')
        train_set, valid_set, test_set = cPickle.load(f)
        f.close()
        self.DSlogger.info("Time loading data into memory: {:.4f}".format(time.time() - t0))

        self.train_in, self.train_obs = self.generate_sequences(train_set)
        self.valid_in, self.valid_obs = self.generate_sequences(valid_set)
        self.test_in, self.test_obs = self.generate_sequences(test_set)
    
    def generate_sequences(self, dset):
        """
        Given training, valid and test sets, create input and observation sets.
        """     
        data_obs = theano.shared(dset[1])
        data_obs = Datapack(data_obs,'vector')

        data_in = dset[0]
        data_in = data_in.reshape((data_in.shape[0],28,28))
        data_in = theano.shared(data_in.swapaxes(0,1).astype(theano.config.floatX))
        data_in = Datapack(data_in,'sequence')

        return [data_in, data_obs]

    def get_train_batches(self, mbsize):
        """
        args:
            - mbsize (int): The size of the minibatch
        """
        return int(self.train_obs.data.get_value().shape[0] // mbsize)

if __name__ == '__main__':

    # mnist = MNISTDataset("./data/mnist.pkl.gz")
    # self.DSlogger.info(mnist.train[0:100].shape)
    char_ds = CharacterDataset("./data/shakespeare.hdf5")
    char_ds.cut_by_sequence(50)
    pdb.set_trace()