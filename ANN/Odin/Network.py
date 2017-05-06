
#### Libraries
# Standard library
import sys
import random
import json

# Third-party libraries
import numpy as np
import theano
import theano.tensor as T

# Odin libraries
from Layers import FullyConnectedLayer
from Layers import ConvolutionPoolLayer
from Layers import SoftmaxLayer

class Network(object):
    def __init__(self, layers, mini_batch_size):
        self.layers = layers
        self.mini_batch_size = mini_batch_size
        self.params = [param for layer in self.layers for param in layer.params]
        self.x = T.matrix('x')
        self.y = T.ivector('y')

        init_layer = self.layers[0]
        init_layer._set_input(self.x, mini_batch_size)

        for j in xrange(1, len(self.layers)):
            prev_layer, layer  = self.layers[j-1], self.layers[j]
            layer._set_input(prev_layer._output, self.mini_batch_size)
        
        self._output = self.layers[-1]._output
    