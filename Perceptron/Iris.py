
import Perceptron as pcp
import Adaline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

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
    for idx, cl in enumerate(np.unique(_target_)):
        plt.scatter(x=_data_[_target_ == cl, 0], y=_data_[_target_ == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)

def main():
    """ Main function """
    #names_ = ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'class']
    df_ = pd.read_csv('iris.data', header=None)

    _target_ = df_.iloc[50:, 4].values
    _target_ = np.where(_target_ == 'Iris-versicolor', 1, -1)
    _data_ = df_.iloc[50:, [1, 3]].values

    _data_std = np.copy(_data_)
    _data_std[:, 0] = (_data_[:, 0] - _data_[:, 0].mean()) / _data_[:, 0].std()
    _data_std[:, 1] = (_data_[:, 1] - _data_[:, 1].mean()) / _data_[:, 1].std()

    train_data = np.concatenate((_data_std[0:40], _data_std[50:90]))
    train_label = np.concatenate((_target_[0:40], _target_[50:90]))

    test_data = np.concatenate((_data_std[40:50], _data_std[90:100]))
    test_label = np.concatenate((_target_[40:50], _target_[90:100]))

    #_classifier = pcp.Perceptron()
    #_classifier = Adaline.AdalineGradientDescent()
    _classifier = Adaline.AdalineStochasticGradientDescent(0.01, 100, random_state=1, shuffle=True)
    _classifier.fit(train_data, train_label)

    plot_decision_regions(test_data, test_label, classifier=_classifier)
    # for x_i, _t_ in zip(test_data, test_label):
    #    res = _classifier.predict(x_i)
    #    if res != _t_:
    #        print 'error'
    #    if res == 1:
    #        plt.scatter(x_i[0], x_i[1], color='red', marker='o')
    #    else:
    #        plt.scatter(x_i[0], x_i[1], color='blue', marker='x')

    plt.tight_layout()
    plt.show()

    
    #plt.plot(_classifier.costs_)
    #plt.show()

if __name__ == "__main__":
    main()
