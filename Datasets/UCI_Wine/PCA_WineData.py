
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler

def main():
    """ Main """
    # Steps in Principle Component Analysis
    # 1. Standarize Data
    # 2. Calculate covariance matrix
    # 3. Decompose covariance matrix into its eigenvectors and eigenvalues
    # 4. Select first k eigenvectors that correspond to first k largest eigenvalues
    # 5. Construct projection matrix W with k eigenvectors
    # 6. Transform feature into new dimensions

    # 1. Get data and standarize it
    df_wine = pd.read_csv('wine.data', header=None)
    df_wine.columns = ['Class label', 'Alcohol',
                       'Malic acid', 'Ash',
                       'Alcalinity of ash', 'Magnesium',
                       'Total phenols', 'Flavanoids',
                       'Nonflavanoid phenols',
                       'Proanthocyanins',
                       'Color intensity', 'Hue',
                       'OD280/OD315 of diluted wines',
                       'Proline']

    feature_list = df_wine.columns[1:]
    _x_ = df_wine.iloc[:, 1:].values # get features
    _y_ = df_wine.iloc[:, 0].values  # get class labels
    x_train, x_test, y_train, y_test = train_test_split(
        _x_,
        _y_,
        test_size=0.3,
        random_state=0
    )
    sc = StandardScaler()
    sc.fit(x_train)
    x_train_std = sc.transform(x_train)
    x_test_std = sc.transform(x_test)

    # 2. Covariance matrix
    cov_mat = np.cov(x_train_std.T)
    eigen_val, eigen_vec = np.linalg.eig(cov_mat)
    #plt.bar(range(len(eigen_val)), eigen_val)
    #plt.xticks(range(len(eigen_val)), feature_list, rotation='vertical')
    #plt.show()

    # 3. Feature Transformation
    eigen_pairs = [(np.abs(eigen_val[i]), eigen_vec[:, i]) for i in range(len(eigen_val))]
    eigen_pairs.sort(reverse=True)
    projection_matrix = np.hstack((eigen_pairs[0][1][:, np.newaxis],
                                   eigen_pairs[1][1][:, np.newaxis])) # select first two pca

    x_train_pca = x_train_std.dot(projection_matrix)
    colors = ['r', 'b', 'g']
    marker = ['s', 'x', 'o']
    for l, c, m in zip(np.unique(y_train), colors, marker):
        plt.scatter(x_train_pca[y_train == l, 0],
                    x_train_pca[y_train == l, 1],
                    c=c,
                    label=l,
                    marker=m)
    plt.show()

    # we can also plot the test data by simplying transforming the x_test_std.

if __name__ == "__main__":
    main()
