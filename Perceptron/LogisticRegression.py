import numpy as np
from numpy.random import seed


class LogisticRegression(object):
    """
    ADAptive LInear NEuron classifier.

    Parameters
    ------------
    _learning_rate : float
        Learning rate (between 0.0 and 1.0)
    _max_iterations : int
        Maximum number of iterations over the training dataset.

    Attributes
    -----------
    __weights__ : 1d-array
        Weights after fitting.
    costs_ : list
        Number of misclassifications (updates) in each epoch.
    """

    def __init__(self, _learning_rate=0.01, _max_iterations=100):
        """ perceptron initialization """
        #self.weights_ = np.random.rand(number_of_features+1)*2-1
        self._learning_rate = _learning_rate
        self._max_iterations = _max_iterations
        self.__weights__ = np.zeros(0)
        self.costs_ = []

    def fit(self, training_data, training_label):
        """
        Fit training data to training_data using training_label

        Parameters
        ----------
        training_data : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        training_label : array-like, shape = [n_samples]
            Target values.
        """

        self.__weights__ = np.random.rand(1 + training_data.shape[1]) * 2 - 1

        for _ in range(self._max_iterations):
            error = 0
            # activation is just a dummy here.
            output = self.__net_input__(training_data)
            error = (training_label - output)
            self.__weights__[1:] += self._learning_rate * training_data.T.dot(error)
            self.__weights__[0] += self._learning_rate * error.sum()
            cost = (error**2).sum() / 2.0
            self.costs_.append(cost)
        return self

    def __net_input__(self, _x_):
        """ Activation Function """
        x = np.dot(_x_, self.__weights__[1:]) + self.__weights__[0]
        return self.__activation__(_x_)

    def __activation__(self, _x_i_):
        """ Activation """
        return self.__sigmoid__(_x_i_)

    def __sigmoid__(self, value):
        """ Sigmoid function """
        return 1.0 / (1.0 + np.exp(-value))

    def predict(self, test_data):
        """
        Returns the class label.
        """
        return np.where(self.__net_input__(test_data) >= 0.5, 1, -1)
