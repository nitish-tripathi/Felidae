
#### Libraries
# Standard library
import sys
import random
import json

# Third-party libraries
import numpy as np
import theano
import theano.tensor as T

class SoftmaxLayer(object):

    def __init__(self, n_in, n_out):
        self.n_in = n_in
        self.n_out = n_out

        # initialize weights and baises
        self.w = theano.shared(value=np.zeros((n_in, n_out), dtype=theano.config.floatX), 
                               name='w',
                               borrow=True)
        
        self.b = theano.shared(value=np.zeros((n_out,), dtype=theano.config.floatX),
                               name='b',
                               borrow=True)
        
        self.params = [self.w, self.b]
    
    def _set_input(self, _input, mini_batch_size):
        self._input = _input.reshape((mini_batch_size, self.n_in))
        self._output =  T.nnet.softmax(T.dot(self._input, self.w) + self.b)
        self.y_out = T.argmax(self._output, axis=1)
    
    def cost(self, net):
        "Return the log-likelihood cost."
        return -T.mean(T.log(self._output)[T.arange(net.y.shape[0]), net.y])

    def accuracy(self, y):
        "Return the accuracy for the mini-batch."
        return T.mean(T.eq(y, self.y_out))