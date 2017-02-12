
import numpy as np
from numpy.random import seed


class AdalineGradientDescent(object):
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
            net_input = self.__net_input__(training_data)

            # activation is just a dummy here.
            output = self.__activation__(training_data)
            error = (training_label - output)
            self.__weights__[1:] += self._learning_rate * training_data.T.dot(error)
            self.__weights__[0] += self._learning_rate * error.sum()
            cost = (error**2).sum() / 2.0
            self.costs_.append(cost)
        return self

    def __net_input__(self, _x_):
        """ Activation Function """
        return np.dot(_x_, self.__weights__[1:]) + self.__weights__[0]

    def __activation__(self, _x_):
        """ Activation """
        return self.__net_input__(_x_)

    def predict(self, test_data):
        """
        Returns the class label.
        """
        return np.where(self.__net_input__(test_data) >= 0, 1, -1)



class AdalineStochasticGradientDescent(object):
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
    shuffle : bool (default: True)
        Shuffles training data every epoch if True to prevent cycles.
    random_state : int (default: None)
        Set random state for shuffling and initializing the weights.
    """

    def __init__(self, _learning_rate=0.01, _max_iterations=100, shuffle=True, random_state=None):
        """ perceptron initialization """
        #self.weights_ = np.random.rand(number_of_features+1)*2-1
        self._learning_rate = _learning_rate
        self._max_iterations = _max_iterations
        self.__weights__ = np.zeros(0)
        self.costs_ = []
        self.shuffle_ = shuffle
        self.w_initialized = False
        if random_state:
            seed(random_state)

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
        self._initialize_weights(training_data.shape[1])
        for _ in range(self._max_iterations):
            if self.shuffle_:
                X, y = self._shuffle(training_data, training_label)
            cost = []
            for x_i, target in zip(X, y):
                cost.append(self._update_weights(x_i, target))
            avg_cost = sum(cost) / len(y)
            self.costs_.append(avg_cost)
        return self

    def partial_fit(self, training_data, training_label):
        """Fit training data without reinitializing the weights"""
        if not self.w_initialized:
            self._initialize_weights(training_data.shape[1])

        # If there is more than one data, then update weights incremently
        # for each of them
        if training_label.ravel().shape[0] > 1:
            for x_i, target_i in zip(training_data, training_label):
                self._update_weights(x_i, target_i)
        else:
            self._update_weights(training_data, training_label)
        return self

    def _initialize_weights(self, shape_):
        """Initialize weights to zeros"""
        self.__weights__ = np.random.rand(1 +shape_) * 2 - 1
        self.w_initialized = True

    def _shuffle(self, _x_, _y_):
        """Shuffle training data"""
        _r_ = np.random.permutation(len(_y_))
        return _x_[_r_], _y_[_r_]

    def _update_weights(self, _x_i_, _target_i_):
        """Apply Adaline learning rule to update the weights"""
        output = self.__net_input__(_x_i_)
        error = (_target_i_ - output)
        self.__weights__[1:] += self._learning_rate * _x_i_.dot(error)
        self.__weights__[0] += self._learning_rate * error
        cost = 0.5 * error**2
        return cost

    def __net_input__(self, _x_):
        """ Activation Function """
        return np.dot(_x_, self.__weights__[1:]) + self.__weights__[0]

    def __activation__(self, _x_):
        """ Activation """
        return self.__net_input__(_x_)

    def predict(self, test_data):
        """
        Returns the class label.
        """
        return np.where(self.__net_input__(test_data) >= 0, 1, -1)

