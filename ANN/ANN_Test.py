
"""
http://neuralnetworksanddeeplearning.com/chap1.html#implementing_our_network_to_classify_digits
http://neuralnetworksanddeeplearning.com/chap2.html
"""

#### Libraries
# Standard library
import sys
import random

# Third-party libraries
import numpy as np
import MNIST_Loader
from sklearn.datasets import make_moons, make_circles

class Network(object):

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def fit(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """
        Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially.
        """

        if test_data: n_test = len(test_data)
        n = len(training_data)

        for j in xrange(epochs):

            # Randomly shuffling training data
            random.shuffle(training_data)

            # Partition training data into mini-batches of the appropriate size
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]

            # Then for each mini_batch we apply a single step of gradient descent
            for mini_batch in mini_batches:
                self.__update_mini_batch__(mini_batch, eta)
            
            if test_data:
                print "Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test)
            else:
                #print "Epoch {0} complete".format(j)
                sys.stderr.write('\rEpoch: %d/%d' % (j+1, epochs))
                sys.stderr.flush()
        print ""

    def __update_mini_batch__(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.__backprop__(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def __backprop__(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.__cost_derivative__(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""

        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)
   
    def __cost_derivative__(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

def main():
    """ Main """
    #training_data, validation_data, test_data = MNIST_Loader.load_data_wrapper()
    #xxx = training_data[0]
    #net = Network([784, 30, 10])
    #net.fit(training_data[:100], 30, 10, 3.0, test_data=test_data[:10])
    
    #X, y = make_moons(200, noise=0.2)
    X, y = make_circles(300, shuffle=True, noise=0.2, factor=0.5)

    # Make sure each input in dataset has the shape (2,1)
    training_inputs = [np.reshape(x, (X.shape[1], 1)) for x in X]

    # Make sure that each result has the shape (2,1)
    y_encoded = one_hot_encoder(y)
    training_results = [np.reshape(x, (y_encoded.shape[1], 1)) for x in y_encoded]

    # Make a tuple of (X, y1)
    training_data = zip(training_inputs, training_results)
        
    # Test data does not have result should not be one-hot-encoded
    test_data = zip(training_inputs, y)

    net = Network([2,4,4,2])
    net.fit(training_data, 300, 1, 1)
    print "Result: {0}/{1}".format(net.evaluate(test_data), len(test_data))

def one_hot_encoder(data):
    create_entry = lambda x : [1, 0] if x == 0 else [0, 1]
    data1 = []
    for x in data:
        e = create_entry(x)
        data1.append(e)
    return np.array(data1)

if __name__ == "__main__":
    main()
