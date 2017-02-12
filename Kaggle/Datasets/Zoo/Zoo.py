import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from pandas.tools.plotting import scatter_matrix
import pandas as pd
import seaborn as sns

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
    """ Main """
    names_ = ['animal_name',
              'hair',
              'feathers',
              'eggs',
              'milk',
              'airborne',
              'aquatic',
              'predator',
              'toothed',
              'backbone',
              'breathes',
              'venomous',
              'fins',
              'legs',
              'tail',
              'domestic',
              'catsize',
              'class_type']
    data_ = pd.read_csv('zoo.csv')
    _x_ = data_.iloc[:, 1:-1].values
    _y_ = data_.iloc[:, -1].values
    
    x_train, x_test, y_train, y_test = train_test_split(_x_, _y_, test_size=0.1, random_state=0)
    sc_ = StandardScaler()
    sc_.fit(x_train)
    x_std = sc_.transform(_x_)
    x_train_std = sc_.transform(x_train)
    x_test_std = sc_.transform(x_test)

    _classifier = SVC(C=10.0, gamma=0.1, kernel="rbf", random_state=0)
    _classifier.fit(x_train_std, y_train)

    for x_i, target in zip(x_test_std, y_test):
        r = _classifier.predict(x_i)
        if r != target:
            print "Error"
        else:
            print "Correct"

    #plot_decision_regions(x_train, y_train, classifier=_classifier)

if __name__ == "__main__":
    main()
