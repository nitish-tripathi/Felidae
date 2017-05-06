
import numpy as np
import theano.tensor as T
import theano
from theano import function
from theano import shared

import Network
import MNIST_Loader
from Layers import ConvolutionPoolLayer, FullyConnectedLayer, SoftmaxLayer

def simple_neuron(x_in, target_in, num_input):
    x = T.fvector('x')
    target = T.fscalar('target')
    W = shared(np.random.rand(2), name='W')
    
    # dot product of x and W
    y = T.dot(x, W)

    # calculate quadratic cost function
    cost = T.sqr(target - y)

    # calculate gradient of cost w.r.t. weights
    gradient = T.grad(cost, [W])
    W_updated = W - (0.1*gradient[0])
    updates = [(W, W_updated)]
    train = function([x, target], y, updates=updates)

    for i in range(20):
        output = train(x_in, target_in)
        print output

def main():
    """ Main """
    #simple_neuron(np.asarray([1.0, 1.0], dtype='float32'), 20, 2)

    training_data, validation_data, test_data = MNIST_Loader.load_data_shared("/Users/nitish/Documents/Python/Machine Learning/ANN/mnist.pkl.gz")
    net = Network.Network([
          ConvolutionPoolLayer.ConvolutionPoolLayer(filter_shape=(20, 1, 5, 5), image_shape=(10, 1, 28, 28), poolsize=(2,2)),
          FullyConnectedLayer.FullyConnectedLayer(n_in=20*12*12, n_out=100),
          SoftmaxLayer.SoftmaxLayer(n_in=100, n_out=10)],
          mini_batch_size=10)
    net.fit(training_data, 60, mini_batch_size=10, eta=0.1, test_data=test_data, validation_data=validation_data)

    #net = ConvolutionPoolLayer.ConvolutionPoolLayer(filter_shape=(2,1,5,5), image_shape=(10, 1, 28, 28), poolsize=(2,2))
    #net = SoftmaxLayer.SoftmaxLayer(100, 10)
    #print net.w.get_value().shape
    #print net.b.get_value().shape

    """
    shared_var = shared(np.array([[0.1, 2],[3, 4]], dtype='float32'))
    x = T.fscalar('x')
    y = x + 2
    updates = [(shared_var, shared_var**2)]
    f1 = function([x], y, updates={shared_var: shared_var * 2})
    print f1(1)
    print shared_var.get_value()
    
    print(theano.config.device)
    print(theano.config.floatX)
    """
if __name__ == "__main__":
    main()