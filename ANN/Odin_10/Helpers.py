
import numpy as np

class Helpers(object):

    @staticmethod
    def sigmoid(z):
        """The sigmoid function."""
        return 1.0/(1.0+np.exp(-z))
    
    @staticmethod
    def sigmoid_prime(z):
        """Derivative of the sigmoid function."""
        return Helpers.sigmoid(z)*(1-Helpers.sigmoid(z))