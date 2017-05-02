
import numpy as np
import theano.tensor as T
from theano import function

def main():
    """ Main """
    #x = T.dmatrix('x')
    #y = T.dmatrix('y')
    #z = x + y
    #f = function([x, y], z)
    #a = np.asarray([[1, 25], [1, 2]])
    #b = np.asarray([[1, 2], [1, 2]])
    #print f(a, b)

    a = T.vector()
    b = T.vector()
    out = a**2 + b**2 + 2*a*b
    f = function([a, b], out)
    print f([1, 2], [1, 2])

if __name__ == "__main__":
    main()