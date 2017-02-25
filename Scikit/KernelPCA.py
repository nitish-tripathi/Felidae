
from scipy.spatial.distance import pdist, squareform
from scipy import exp
from sklearn.datasets import make_moons, make_circles
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

def rbf_kernel(_x_, gamma, n_components):
    """
       RBF kernel PCA implementation.
       Parameters
       ------------
       _x_: {NumPy ndarray}, shape = [n_samples, n_features]
       gamma: float
         Tuning parameter of the RBF kernel
       n_components: int
         Number of principal components to return
       Returns
       ------------
        X_pc: {NumPy ndarray}, shape = [n_samples, k_features]
          Projected dataset
    """

    sq_dists = pdist(_x_, 'sqeuclidean')
    mat_sq_dists = squareform(sq_dists)

    # Computing the MxM kernel matrix.
    K = exp(-gamma * mat_sq_dists)

    # Centering the symmetric NxN kernel matrix.
    N = K.shape[0]
    one_n = np.ones((N, N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

    # Obtaining eigenvalues in descending order with corresponding
    # eigenvectors from the symmetric matrix.
    eigvals, eigvecs = np.linalg.eigh(K)

    # Obtaining the i eigenvectors that corresponds to the i highest eigenvalues.
    x_pc = np.column_stack((eigvecs[:, -i] for i in range(1, n_components+1)))

    return x_pc


def main():
    """ Main """
    #_x_, _y_ = make_moons(n_samples=200, random_state=123, noise=0.2)
    _x_, _y_ = make_circles(n_samples=200, random_state=123, noise=0.1, factor=0.2)

    # Implementing using PCA
    pca = PCA(n_components=2)
    x_pca = pca.fit_transform(_x_)
    x_rbf = rbf_kernel(_x_, 15, 2)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    ax[0].scatter(x_pca[_y_ == 0, 0], x_pca[_y_ == 0, 1], color='red')
    ax[0].scatter(x_pca[_y_ == 1, 0], x_pca[_y_ == 1, 1], color='blue')
    ax[0].set_title("Using PCA")

    ax[1].scatter(x_rbf[_y_ == 0, 0], x_rbf[_y_ == 0, 1], color='red')
    ax[1].scatter(x_rbf[_y_ == 1, 0], x_rbf[_y_ == 1, 1], color='blue')
    ax[1].set_title("Using Kernel PCA")

    plt.show()

if __name__ == '__main__':
    main()
