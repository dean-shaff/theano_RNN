import time 
import pdb
import cPickle
import gzip

import numpy as np 
import theano 
import theano.tensor as T 

class Datapack(object):
    """
    A class that represents individual data units 
    eg training input data, training observation data. 
    """
    def __init__(self, data,dtype):
        self.data = data 
        self.dtype = dtype

    def __getitem__(self,ind):
        if self.dtype == 'sequence':
            return self.data[:,ind,:]
        elif self.dtype == 'vector':
            return self.data[ind]
 
class MNISTDataset(object):
    """
    A class that contains the training, testing and validation sets for MNIST.
    """
    def __init__(self, filename):
        """
        args:
            - filename (str): The filename of the MNIST pickle file.
        """    
        print("Loading data into memory")
        t0 = time.time()
        f = gzip.open(filename, 'rb')
        train_set, valid_set, test_set = cPickle.load(f)
        f.close()
        print("Time loading data into memory: {:.4f}".format(time.time() - t0))

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

    mnist = MNISTDataset("./data/mnist.pkl.gz")
    # print(mnist.train[0:100].shape)
    pdb.set_trace()