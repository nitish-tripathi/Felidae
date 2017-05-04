
import numpy as np
import theano.tensor as T
from theano import function
from theano import shared

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
    simple_neuron(np.asarray([1.0, 1.0], dtype='float32'), 20, 2)

if __name__ == "__main__":
    main()