
"""
http://neuralnetworksanddeeplearning.com/chap1.html#implementing_our_network_to_classify_digits
http://neuralnetworksanddeeplearning.com/chap2.html
http://numericinsight.com/uploads/A_Gentle_Introduction_to_Backpropagation.pdf
"""

#### Libraries
# Standard library
import sys
import random
import cPickle

# Third-party libraries
import numpy as np

class QuadraticCost(object):
    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.
        """
        return 0.5*np.linalg.norm(a-y)**2

    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer."""
        return (a-y) * sigmoid_prime(z)

class CrossEntropyCost(object):
    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output ``y``.  
        Note that np.nan_to_num is used to ensure numerical
        stability.  In particular, if both ``a`` and ``y`` have a 1.0
        in the same slot, then the expression (1-y)*np.log(1-a)
        returns nan.  The np.nan_to_num ensures that that is converted
        to the correct value (0.0).
        """
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))
    
    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer.  
        Note that theparameter ``z`` is not used by the method.  It is included in
        the method's parameters in order to make the interface
        consistent with the delta method for other cost classes.
        """
        return (a-y)

class Network(object):

    def __init__(self, sizes, cost=CrossEntropyCost):
        """
        Initializes artificial neural network classifier.
        
        The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers.

        Parameters
        ---------
        sizes: 1d array
            Contains the number of neurons in the respective layers of 
            the network. For example, if the list was [2, 3, 1] then it 
            would be a three-layer network, with the first layer containing 
            2 neurons, the second layer 3 neurons, and the third layer 1 neuron.
        
        cost: Cost class
            Defines the cost calculation class, either CrossEntropyCost or
            Quadratic cost
        """

        self._num_layers = len(sizes)
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        self.cost = cost

    def _feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a
    
    def _feedforward2(self, a):
        zs = []
        activations = [a]

        activation = a
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        return (zs, activations)

    def fit(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """
        Fit the model to the training data.
        
        Train the neural network using mini-batch stochastic
        gradient descent.

        Parameters
        ---------
        training_data: list of tuples (X, y)
            X is input and y is desired output
        
        epoch: int
            Maximum number of iterations over the training dataset.
        
        mini_batch_size: int
            Compute gradient against more than one training example at
            each step.
        
        eta: float
            Learning rate
        
        test_data: list of tuples (X, y)
            If provided then the network will be evaluated against the 
            test data after each epoch, and partial progress printed out.
            This is useful for tracking progress, but slows things down 
            substantially.
        
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
                #self._update_mini_batch_old(mini_batch, eta)
                self._update_mini_batch(mini_batch, eta)
            
            if test_data:
                print "Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test)
            else:
                #print "Epoch {0} complete".format(j)
                sys.stderr.write('\rEpoch: %d/%d' % (j+1, epochs))
                sys.stderr.flush()
        print ""

    def _update_mini_batch_old(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self._backpropagation(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def _update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""

        batch_size = len(mini_batch)

        # transform to (input x batch_size) matrix
        x = np.asarray([_x.ravel() for _x, _y in mini_batch]).transpose()
        # transform to (output x batch_size) matrix
        y = np.asarray([_y.ravel() for _x, _y in mini_batch]).transpose()

        nabla_b, nabla_w = self._backpropagation(x, y)
        self.weights = [w - (eta / batch_size) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / batch_size) * nb for b, nb in zip(self.biases, nabla_b)]
        return

    def _backpropagation(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""

        nabla_b = [0 for i in self.biases]
        nabla_w = [0 for i in self.weights]

        # feedforward
        zs, activations = self._feedforward2(x)

        # backward pass
        delta = self.cost.delta(zs[-1], activations[-1], y)
        #delta = self._cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta.sum(1).reshape([len(delta), 1]) # reshape to (n x 1) matrix
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in xrange(2, self._num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta.sum(1).reshape([len(delta), 1]) # reshape to (n x 1) matrix
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return (nabla_b, nabla_w)

    def _backpropagation_old(self, x, y):
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
        delta = self._cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in xrange(2, self._num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """
        Evaluate the test data.

        Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation.
        
        Parameters
        ---------
        test_data: list of tuples (X, y)
            X is input and y is desired output
        """

        test_results = [(np.argmax(self._feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)
   
    def _cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

    def save(self, filename="model"):
        """ Method to save the trained model"""
        f = open(filename, 'wb')
        cPickle.dump(self, f, protocol=-1)
        f.close()

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
