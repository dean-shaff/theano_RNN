import time
import pdb

import numpy as np 
import theano 
import theano.tensor as T 

class LSTMLayer(object):

    def __init__(self,X,dim,**kwargs):
        """
        Set up the weight matrices for a long short term memory (LSTM) unit. 
        I use the notation from Graves. 
        args:
            - dim: A dictionary containing the dimensions of the units inside the LSTM.  
        kwargs:
            - 
        """
        uni = np.random.uniform

        def diag_constructor(limit,size):
            """
            args:
                - limit: A list whose two elements correspond to the limit for the numpy uniform function.
                - size: (Int) one dimension of the square matrix.
            """
            diag_val = uni(limit[0], limit[1],size)
            diag_ind = np.diag_indices(size)
            mat = np.zeros((size,size))
            mat[diag_ind] = diag_val
            return mat.astype(theano.config.floatX)          


        truncate = kwargs.get("bptt_truncate", -1)

        nin = dim.get('in_dim')
        nout = dim.get('out_dim')
        nhid = dim.get('hid_dim')
        self.nin = nin
        self.nout = nout 
        self.nhid = nhid 
        # print("hidden dim", nhid)

        # Input
        self.WiL = theano.shared(uni(-np.sqrt(1.0/(nin)), np.sqrt(1.0/(nin)),(nin, nhid)).astype(theano.config.floatX),name='WiL')
        self.WhL = theano.shared(uni(-np.sqrt(1.0/(nhid)), np.sqrt(1.0/(nhid)),(nhid, nhid)).astype(theano.config.floatX),name='WhL')
        self.WcL = theano.shared(diag_constructor([-np.sqrt(1.0/(nhid)), np.sqrt(1.0/(nhid))],nhid),name='WcL')
        self.bL = theano.shared(np.zeros(nhid),name='bL')
        # Forget
        self.WiPhi = theano.shared(uni(-np.sqrt(1.0/(nin)), np.sqrt(1.0/(nin)),(nin, nhid)).astype(theano.config.floatX),name='WiPhi')
        self.WhPhi = theano.shared(uni(-np.sqrt(1.0/(nhid)), np.sqrt(1.0/(nhid)),(nhid, nhid)).astype(theano.config.floatX),name='WhPhi')
        self.WcPhi = theano.shared(diag_constructor([-np.sqrt(1.0/(nhid)), np.sqrt(1.0/(nhid))],nhid),name='WcL')
        self.bPhi = theano.shared(np.zeros(nhid),name='bPhi')
        # Cells
        self.WiCell = theano.shared(uni(-np.sqrt(1.0/(nin)), np.sqrt(1.0/(nin)),(nin, nhid)).astype(theano.config.floatX),name='WiCell')
        self.WhCell = theano.shared(uni(-np.sqrt(1.0/(nhid)), np.sqrt(1.0/(nhid)),(nhid, nhid)).astype(theano.config.floatX),name='WhCell')
        self.bCell = theano.shared(np.zeros(nhid),name='bCell')
        # Output
        self.WiOm = theano.shared(uni(-np.sqrt(1.0/(nin)), np.sqrt(1.0/(nin)),(nin, nhid)).astype(theano.config.floatX),name='WiOm')
        self.WhOm = theano.shared(uni(-np.sqrt(1.0/(nhid)), np.sqrt(1.0/(nhid)),(nhid, nhid)).astype(theano.config.floatX),name='WhOm')
        self.WcOm = theano.shared(diag_constructor([-np.sqrt(1.0/(nhid)), np.sqrt(1.0/(nhid))],nhid),name='WcL')
        self.bOm = theano.shared(np.zeros(nhid),name='bOm')
        # Sequence output 
        self.Wy = theano.shared(uni(-np.sqrt(1.0/(nhid)), np.sqrt(1.0/(nhid)),(nhid,nout)).astype(theano.config.floatX),name='Wy')
        self.by = theano.shared(np.zeros(nout), name='by')

        self.params = [
                        self.WiL, self.WhL, self.WcL,self.bL,
                        self.WiPhi, self.WhPhi, self.WcPhi,self.bPhi,
                        self.WiCell, self.WhCell,self.bCell,
                        self.WiOm, self.WhOm, self.WcOm,self.bOm,
                        self.Wy, self.by
                    ]
        def recurrent_step(x_t,b_tm1,s_tm1):
            """
            Define the recurrent step.
            args:
                - x_t: the current sequence
                - b_tm1: the previous b_t (b_{t minus 1})
                - s_tml: the previous s_t (s_{t minus 1}) this is the state of the cell
            """
            # Input 
            b_L = T.nnet.sigmoid(T.dot(x_t, self.WiL) + T.dot(b_tm1,self.WhL) + T.dot(s_tm1, self.WcL) + self.bL)
            # Forget
            b_Phi = T.nnet.sigmoid(T.dot(x_t,self.WiPhi) + T.dot(b_tm1,self.WhPhi) + T.dot(s_tm1, self.WcPhi) + self.bPhi)
            # Cell 
            a_Cell = T.dot(x_t, self.WiCell) + T.dot(b_tm1, self.WhCell) + self.bCell
            s_t = b_Phi * s_tm1 + b_L*T.tanh(a_Cell)
            # Output 
            b_Om = T.nnet.sigmoid(T.dot(x_t, self.WiOm) + T.dot(b_tm1,self.WhOm) + T.dot(s_t, self.WcOm) + self.bOm)
            # Final output (What gets sent to the next step in the recurrence) 
            b_Cell = b_Om*T.tanh(s_t)
            # Sequence output
            o_t = T.nnet.softmax(T.dot(b_Cell, self.Wy) + self.by)

            return b_Cell, s_t, o_t 

        out, _ = theano.scan(recurrent_step,
                                truncate_gradient=truncate,
                                sequences = X,
                                outputs_info=[
                                                {'initial':T.zeros((X.shape[1],nhid))},
                                                {'initial':T.zeros((X.shape[1],nhid))},
                                                {'initial':None}
                                            ],
                                n_steps=X.shape[0])

        self.b_out = out[0]
        self.pred = out[2]

if __name__ == "__main__":
    x = T.tensor3('x')
    lstm = LSTMLayer(x,{'in_dim':100,'hid_dim':100,'out_dim':20})
    t0 = time.time()
    f = theano.function([x],lstm.pred)
    print("Took {:.4f} seconds to compile".format(time.time() - t0))
    x0 = np.random.randn(150,50,100) #sequence length,  batch size, character length (input length)
    res = f(x0)
    pdb.set_trace()