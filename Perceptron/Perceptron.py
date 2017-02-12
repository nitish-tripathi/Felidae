
import numpy as np

class Generate2dData(object):
    """ Class to generate demo 2d data"""

    def __init__(self, number_of_samples):
        """ Initializer """
        self.number_of_samples = number_of_samples

    def generate(self):
        """
        generates a 2D linearly separable dataset with n samples.
        The third element of the sample is the label
        """
        x_b = (np.random.rand(self.number_of_samples)*2-1)/2-0.5
        y_b = (np.random.rand(self.number_of_samples)*2-1)/2+0.5
        x_r = (np.random.rand(self.number_of_samples)*2-1)/2+0.5
        y_r = (np.random.rand(self.number_of_samples)*2-1)/2-0.5

        inputs = []
        labels = []
        for i, _ in enumerate(x_b):
            inputs.append([x_b[i], y_b[i]])
            labels.append(1)
            inputs.append([x_r[i], y_r[i]])
            labels.append(-1)

        return inputs, labels

class Perceptron(object):
    """
    Perceptron Classifier
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
    errors_ : list
        Number of misclassifications (updates) in each epoch.
    """

    def __init__(self, _learning_rate=0.1, _max_iterations=100):
        """ perceptron initialization """
        #self.weights_ = np.random.rand(number_of_features+1)*2-1
        self._learning_rate = _learning_rate
        self._max_iterations = _max_iterations
        self.__weights__ = np.zeros(0)
        self.errors_ = []

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

        self.__weights__ = np.random.rand(1+training_data.shape[1])*2-1
        for _ in range(self._max_iterations):
            error = 0
            for x_i, target in zip(training_data, training_label):
                update = self._learning_rate * (target - self.predict(x_i))
                self.__weights__[1:] += update * x_i
                self.__weights__[0] += update
                error += int(update != 0.0)
            self.errors_.append(error)
        return self

    def __net_input__(self, _x_):
        """ Activation Function """
        dot_product = np.dot(_x_, self.__weights__[1:]) + self.__weights__[0]
        return np.where(dot_product >= 0, 1, -1)

    def predict(self, test_data):
        """
        Returns the class label.
        """
        return self.__net_input__(test_data)
