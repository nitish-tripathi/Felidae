
import numpy as np

def softmax(X):
    """ Softmax function """
    return np.exp(X) / np.sum(np.exp(X), axis=0)

def main():
    """ Main """
    scores = np.array([1.0, 2.0, 3.0])
    print softmax(scores)

    scores = np.array([[1, 2, 3, 6],
                      [2, 4, 5, 6],
                      [3, 8, 7, 6]])
    print softmax(scores)


if __name__ == "__main__":
    main()