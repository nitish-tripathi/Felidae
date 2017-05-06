
"""
http://neuralnetworksanddeeplearning.com/chap6.html
"""

#### Libraries
# Standard library
import sys
import random

# Third-party libraries
import numpy as np
import theano
import theano.tensor as T

class FullyConnectedLayer(object):
    """ Class that implements fully connected layer """

    def __init__(self, n_in, n_out):
        self.n_in = n_in
        self.n_out = n_out
        self.activation_fn = T.nnet.sigmoid

        # initialize weights
        self.w = theano.shared(np.asarray(np.random.normal(
                                          loc=0.0,
                                          scale=np.sqrt(1.0/n_out),
                                          size=(n_in, n_out)), dtype=theano.config.floatX),
                                name='w', 
                                borrow=True)
        self.b = theano.shared(np.asarray(np.ones((n_out,)), dtype=theano.config.floatX),
                                name='b', borrow=True)
        
        self.params = [self.w, self.b]
    
    def _set_input(self, _input, mini_batch_size):
        self._input = _input.reshape((mini_batch_size, self.n_in))
        self._output = self.activation_fn(T.dot(self._input, self.w) + self.b)
        self.y_out = T.argmax(self._output, axis=1)
    
    def accuracy(self, y):
        """Return the accuracy for the mini-batch."""
        return T.mean(T.eq(y, self.y_out))

