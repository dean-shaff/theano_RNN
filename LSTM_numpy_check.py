"""
The purpose of this script is to check whether theano (really just theano.scan) 
is doing what I expect it to.
"""
from __future__ import print_function
import time 
import pdb

import numpy as np 
import theano 
import theano.tensor as T 

from LSTM import LSTMLayer 

x = T.tensor3('x') 

lstm = LSTMLayer(x,{'in_dim':50,'hid_dim':100,'out_dim':20})
WiL = lstm.WiL.get_value()
WhL = lstm.WhL.get_value()
WcL = lstm.WcL.get_value()
# Forget
WiPhi = lstm.WiPhi.get_value()
WhPhi = lstm.WhPhi.get_value()
WcPhi = lstm.WcPhi.get_value()
# Cells
WiCell = lstm.WiCell.get_value()
WhCell = lstm.WhCell.get_value()
# Output
WiOm = lstm.WiOm.get_value()
WhOm = lstm.WhOm.get_value()
WcOm = lstm.WcOm.get_value()
# sequence output 
Wy = lstm.Wy.get_value()
by = lstm.by.get_value()

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def recurrent_step(x_t,b_tm1,s_tm1):
    """
    Define the recurrent step.
    args:
        - x_t: the current sequence
        - b_tm1: the previous b_t (b_{t minus 1})
        - s_tml: the previous s_t (s_{t minus 1}) this is the state of the cell
    """
    # Input 
    b_L = np.tanh(np.dot(x_t, WiL) + np.dot(b_tm1,WhL) + np.dot(s_tm1, WcL))
    # Forget
    b_Phi = np.tanh(np.dot(x_t,WiPhi) + np.dot(b_tm1,WhPhi) + np.dot(s_tm1, WcPhi))
    # Cell 
    a_Cell = np.dot(x_t, WiCell) + np.dot(b_tm1, WhCell)
    s_t = b_Phi * s_tm1 + b_L*np.tanh(a_Cell)
    # Output 
    b_Om = np.tanh(np.dot(x_t, WiOm) + np.dot(b_tm1,WhOm) + np.dot(s_t, WcOm))
    # Final output (What gets sent to the next step in the recurrence) 
    b_Cell = b_Om*np.tanh(s_t)
    # sequence output
    o_t = sigmoid(np.dot(b_Cell, Wy) + by)

    return b_Cell, s_t, o_t

batch = 75
x0 = np.random.randn(150,batch,lstm.nin)
t0 = time.time()
res_np = np.zeros((150,batch,lstm.nout))
b, s, o = recurrent_step(x0[0],np.zeros((batch,lstm.nhid)),np.zeros((batch,lstm.nhid)))
res_np[0,:,:] = o
for i in xrange(1,res_np.shape[0]):
    b,s,o = recurrent_step(x0[i],b,s)
    res_np[i,:,:] = o
print("Numpy calculation time: {:.3f}".format(time.time() - t0))

print("Compiling theano function...")
f_th = theano.function([x],lstm.pred)
print("Done compiling")

t0 = time.time()
res_th = f_th(x0)
print("Theano calculation time: {:.3f}".format(time.time() - t0))

print("Are the two results the same?: {}".format(np.allclose(res_th, res_np)))