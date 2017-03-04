
import matplotlib.pylab as plt
import numpy as np
import Perceptron

# initialize perceptron for 2 features input vector
__perceptron__ = Perceptron.Perceptron()

#__trainset__, __train_labels__ = Perceptron.Generate2dData(30).generate()
__trainset__ = [
    np.array([0, 0]),
    np.array([1, 0]),
    np.array([0, 1]),
    np.array([1, 1])
    ]
__train_labels__ = [-1, 1, 1, 1]

__perceptron__.fit(np.array(__trainset__), np.array(__train_labels__))

#__testset__, __test_labels__ = Perceptron.Generate2dData(20).generate()  # test set generation
__testset__ = [
    np.array([0, 0]),
    np.array([1, 0]),
    np.array([0, 1]),
    np.array([1, 1])
    ]
__test_labels__ = [-1, 1, 1, 1]

# Perceptron test
for x, t in zip(__testset__, __test_labels__):
    r = __perceptron__.predict(x)
    #print "{}: {}".format(x[:-1], r)

    if r != t:
        print 'error'
    if r == 1:
        plt.scatter(x[0], x[1], color='red', marker='o')
    else:
        plt.scatter(x[0], x[1], color='blue', marker='x')

plt.show()
