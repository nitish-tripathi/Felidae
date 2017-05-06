
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
        self.init_layer = self.layers[0]
    