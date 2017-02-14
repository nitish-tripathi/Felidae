
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

class Helpers(object):
    """ Helpers class """

    def __init__(self):
        """ Default initializer for Helpers class"""

    def plot_decision_regions(self, _data_, _target_, classifier, resolution=0.02):
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

    def generate_random_2d_data(self, number_of_samples=200):
        """
        generates a 2D linearly separable dataset with n samples.
        The third element of the sample is the label
        """
        x_b = (np.random.rand(number_of_samples)*2-1)/2-0.5
        y_b = (np.random.rand(number_of_samples)*2-1)/2+0.5
        x_r = (np.random.rand(number_of_samples)*2-1)/2+0.5
        y_r = (np.random.rand(number_of_samples)*2-1)/2-0.5

        inputs = []
        labels = []
        for i, _ in enumerate(x_b):
            inputs.append([x_b[i], y_b[i]])
            labels.append(1)
            inputs.append([x_r[i], y_r[i]])
            labels.append(-1)

        return inputs, labels
