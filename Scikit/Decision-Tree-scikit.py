
"""
Decision tree implementation using scikit library
"""

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split

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

def main():
    """ Main """
     # Iris dataset
    #iris = datasets.load_iris()
    #_x_ = iris.data[:, [2, 3]]
    #_y_ = iris.target[:]

    # Xor data
    np.random.seed(0)
    _x_ = np.random.randn(400, 2)
    _y_ = np.logical_xor(_x_[:, 0] > 0, _x_[:, 1] > 0)
    _y_ = np.where(_y_, 1, -1)

    x_train, x_test, y_train, y_test = train_test_split(_x_, _y_, test_size=0.1, random_state=0)
    sc_ = StandardScaler()
    sc_.fit(x_train)
    x_std = sc_.transform(_x_)
    x_train_std = sc_.transform(x_train)
    x_test_std = sc_.transform(x_test)

    _classifier = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=None)
    _classifier.fit(x_train_std, y_train)

    plot_decision_regions(x_std, _y_, classifier=_classifier)

    # To convert dot file to png use
    # dot -Tpng tree.dot -o tree.png
    export_graphviz(_classifier, out_file="tree.dot")
    plt.show()

if __name__ == "__main__":
    main()
