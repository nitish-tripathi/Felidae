
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles

from Odin import Network, MNIST_Loader

def main():
    """ Main """
    
    training_data, validation_data, test_data = MNIST_Loader.load_data_wrapper()
    #xxx = training_data[0]
    net = Network.Network([784, 100, 10], eta=0.5, C=5)
    net.fit(training_data, 30, 10, test_data=test_data[:10000], calc_test_cost=True)
    
    plt.plot(net.test_cost, '-')
    plt.show()

    """
    #X, y = make_moons(200, noise=0.2)
    X, y = make_circles(200, shuffle=True, noise=0.2, factor=0.5)

    # Make sure each input in dataset has the shape (2,1)
    training_inputs = [np.reshape(x, (X.shape[1], 1)) for x in X]

    # Make sure that each result has the shape (2,1)
    y_encoded = one_hot_encoder(y)
    training_results = [np.reshape(x, (y_encoded.shape[1], 1)) for x in y_encoded]

    # Make a tuple of (X, y1)
    training_data = zip(training_inputs, training_results)
        
    # Test data does not have result should not be one-hot-encoded
    test_data = zip(training_inputs, y)

    net = Network.Network([2,3,2], eta=0.1, C=0.01)
    net.fit(training_data, 300, 1, test_data=test_data, calc_test_cost=True)
    print "Result: {0}/{1}".format(net.evaluate(test_data), len(test_data))
    
    plt.plot(net.test_cost, '-')
    plt.show()
    """

def one_hot_encoder(data):
    create_entry = lambda x : [1, 0] if x == 0 else [0, 1]
    data1 = []
    for x in data:
        e = create_entry(x)
        data1.append(e)
    return np.array(data1)

if __name__ == "__main__":
    main()