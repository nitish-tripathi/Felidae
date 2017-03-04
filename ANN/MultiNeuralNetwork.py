
import numpy as np
from sklearn.datasets import make_moons, make_circles
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class MultiNeuralNetwork(object):
    """ Multiple Neural Network Implementation """

    def __init__(self, num_outputs=1, hidden_layer=3, max_iter=20000, _learning_rate=0.01, C=0.01):
        """ Initialize """
        self._learning_rate = _learning_rate
        self.__weights__ = np.zeros(0)
        self._hidden_layer = hidden_layer
        self.max_iter = max_iter
        self.model = {}
        self.num_outputs = num_outputs
        self.C = C

    def predict(self, X):
        """
        Helper function to predict an output (0 or 1)
        """
        W1, b1, W2, b2 = self.model['W1'], self.model['b1'], self.model['W2'], self.model['b2']
        # Forward propagation
        z1 = X.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return np.argmax(probs, axis=1)

    def fit(self, X, y):      
        """
        This function learns parameters for the neural network and returns the model.
        - nn_hdim: Number of nodes in the hidden layer
        - num_passes: Number of passes through the training data for gradient descent
        - print_loss: If True, print the loss every 1000 iterations
        """
        num_examples = len(X) # training set size
        nn_input_dim = X.shape[1] # input layer dimensionality

        # Initialize the parameters to random values. We need to learn these.
        np.random.seed(0)
        self.W1 = np.random.randn(nn_input_dim, self._hidden_layer) / np.sqrt(nn_input_dim)
        self.b1 = np.zeros((1, self._hidden_layer))
        self.W2 = np.random.randn(self._hidden_layer, self.num_outputs) / np.sqrt(self._hidden_layer)
        self.b2 = np.zeros((1, self.num_outputs))
        
        # Gradient descent. For each batch...
        for i in xrange(0, self.max_iter):

            # Forward propagation
            z1 = X.dot(self.W1) + self.b1
            a1 = np.tanh(z1)
            z2 = a1.dot(self.W2) + self.b2
            exp_scores = np.exp(z2)
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

            # Backpropagation
            delta3 = probs
            delta3[range(num_examples), y] -= 1
            dW2 = (a1.T).dot(delta3)
            db2 = np.sum(delta3, axis=0, keepdims=True)
            delta2 = delta3.dot(self.W2.T) * (1 - np.power(a1, 2))
            dW1 = np.dot(X.T, delta2)
            db1 = np.sum(delta2, axis=0)

            # Add regularization terms (b1 and b2 don't have regularization terms)
            dW2 += self.C * self.W2
            dW1 += self.C * self.W1

            # Gradient descent parameter update
            self.W1 += -self._learning_rate * dW1
            self.b1 += -self._learning_rate * db1
            self.b1 += -self._learning_rate * db1
            self.W2 += -self._learning_rate * dW2
            self.b2 += -self._learning_rate * db2
            
            # Assign new parameters to the model
            self.model = { 'W1': self.W1, 'b1': self.b1, 'W2': self.W2, 'b2': self.b2}

def main():
    """ main """
    # Generate moon data
    np.random.seed(0)
    X, y = make_moons(200, noise=0.2)
    clf_nn = MultiNeuralNetwork(num_outputs=2)
    clf_nn.fit(X, y)
    plot_decision_regions(X, y, classifier=clf_nn)
    plt.show()
    
def plot_decision_regions(_data_, _target_, classifier, resolution=0.02):
        """ Plot decision boundary mesh """
        # setup marker generator and color map
        markers = ('s', 'x', 'o', '^', 'v')
        colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
        cmap = ListedColormap(colors[:len(np.unique(_target_))])

        # plot the decision surface
        x1_min, x1_max = _data_[:, 0].min() - 1, _data_[:, 0].max() + 1
        x2_min, x2_max = _data_[:, 1].min() - 1, _data_[:, 1].max() + 1
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                               np.arange(x2_min, x2_max, resolution))
        predicted_class = classifier.predict(
            np.array([xx1.ravel(), xx2.ravel()]).T)
        predicted_class = predicted_class.reshape(xx1.shape)
        plt.contourf(xx1, xx2, predicted_class, alpha=0.4, cmap=cmap)
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())

        # plot class samples
        for idx, classy_ in enumerate(np.unique(_target_)):
            plt.scatter(x=_data_[_target_ == classy_, 0], y=_data_[_target_ == classy_, 1],
                        alpha=0.8, c=cmap(idx),
                        marker=markers[idx], label=classy_)


if __name__ == '__main__':
    main()