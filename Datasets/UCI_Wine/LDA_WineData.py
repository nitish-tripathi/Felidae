
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler

def main():
    """ Main """
    df_wine = pd.read_csv('wine.data', header=None)
    _x_ = df_wine.iloc[:, 1:].values
    _y_ = df_wine.iloc[:, 0].values

    # 1. Standarize data
    x_train, x_test, y_train, y_test = train_test_split(_x_, _y_, test_size=0.2, random_state=0)
    sc_ = StandardScaler()
    sc_.fit(x_train)
    x_train_std = sc_.transform(x_train)
    x_test_std = sc_.transform(x_test)

    # 2. Calculate mean of features for every class
    np.set_printoptions(precision=4)
    mean_vecs = []
    for label in np.unique(_y_):
        mean_vecs.append(
            np.mean(x_train_std[y_train == label], axis=0)
        )
    _d_ = _x_.shape[1] # number of features
    _sw_ = np.zeros((_d_, _d_))
    for label in np.unique(_y_):
        class_scatter = np.cov(x_train_std[y_train == label].T)
        _sw_ += class_scatter
    print 'Within-class scatter matrix: %sx%s' % (_sw_.shape[0], _sw_.shape[1])

    mean_overall = np.mean(x_train_std, axis=0)
    _sb_ = np.zeros((_d_, _d_))
    for i, mean_vec in enumerate(mean_vecs):
        n = _x_[_y_ == i+1, :].shape[0]
        mean_vec = mean_vec.reshape(_d_, 1)
        mean_overall = mean_overall.reshape(_d_, 1)
    _sb_ += n * (mean_vec - mean_overall).dot((mean_vec - mean_overall).T)
    print 'Between-class scatter matrix: %sx%s' % (_sb_.shape[0], _sb_.shape[1])

    # 3. Calculate eigenvectors and eigenvalues
    eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(_sw_).dot(_sb_))

    eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, 1]) for i in range(len(eigen_vals))]
    eigen_pairs = sorted(eigen_pairs, key=lambda k: k[0], reverse=True)
    for val in eigen_pairs:
        print val[0]

if __name__ == "__main__":
    main()
