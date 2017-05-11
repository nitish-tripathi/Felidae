
"""
Baby steps - Algebra
"""

import numpy as np
import theano.tensor as T
from theano import function
from theano import pp

def main():
    x = T.dscalar('x')
    y = T.dscalar('y')
    z = x + y
    f = function([x, y], z)
    print z.eval({x:1,y:2})
    print pp(z)
    print f(1,2)

    a = T.vector()
    out = a + a**10
    f1 = function([a], out)
    print f1([0, 1, 2])

    a = T.vector()
    b = T.vector()
    out = a**2 + 2*a*b + b**2
    f3 = function([a,b], out)
    print f3([1,2],[2,1])

if __name__ == "__main__":
    main()