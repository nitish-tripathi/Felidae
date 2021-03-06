
import os
import struct
import numpy as np
import matplotlib.pyplot as plt
from neuralnet import NeuralNetMLP
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def load_mnist(kind='train'):
    """ Load MNIST data """
    labels_path = kind + '-labels-idx1-ubyte'
    images_path = kind + '-images-idx3-ubyte'

    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)
    
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII",
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)
                
    return images, labels

def main():
    """ Main """
    # Load Data
    X_train, y_train = load_mnist()
    X_test, y_test = load_mnist(kind='t10k')

    print "Training Images- Rows: %d, Columns: %d" % (X_train.shape[0], X_train.shape[1])
    print "Training Labels- Rows: %d" % (y_train.shape[0])

    print "Test Images- Rows: %d, Columns: %d" % (X_test.shape[0], X_test.shape[1])
    print "Test Labels- Rows: %d" % (y_test.shape[0])

    """
    # Show one sample for every type of digit
    fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True)
    ax = ax.flatten()
    for i in range(10):
        img = X_train[y_train == i][0].reshape(28,28)
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')
    ax[0].set_xticks([])    # Don't show ticks on plot
    ax[0].set_yticks([])
    plt.show()

    # Show different types of 8
    fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True)
    ax = ax.flatten()
    for i in range(25):
        img = X_train[y_train == 8][i].reshape(28,28)
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.show()
    """

    nn = NeuralNetMLP(n_output=10, 
                  n_features=X_train.shape[1], 
                  n_hidden=50, 
                  l2=0.1, 
                  l1=0.0, 
                  epochs=1000, 
                  eta=0.001,
                  alpha=0.001,
                  decrease_const=0.00001,
                  minibatches=50, 
                  shuffle=True,
                  random_state=1)

    nn.fit(X_train, y_train, print_progress=True)
    nn.save()
    y_train_pred = nn.predict(X_train)

    acc = ((np.sum(y_train == y_train_pred, axis=0)).astype('float') /
           X_train.shape[0])

    print "Accuracy: %0.2f" % acc

if __name__ == '__main__':
    main()
