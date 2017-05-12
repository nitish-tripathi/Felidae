
import numpy as np
import theano.tensor as T
import theano

def main():
    """ main """

    # Logistic functions
    x = T.dmatrix('x')
    s = 1 / (1 + T.exp(-x))
    s2 = (1 + T.tanh(x/2)) / 2

    logistic = theano.function([x], s)
    logistic2 = theano.function([x], s2)

    print logistic([[0, 1], [-1, -2]])
    print logistic2([[0, 1],[-1, -2]])

    # Functions with more than one input and output
    a, b = T.dmatrices('a', 'b')
    diff = a - b
    abs_diff = abs(diff)
    diff_sqr = abs_diff**2
    f = theano.function([a, b], [diff, abs_diff, diff_sqr])
    print f([[1, 0], [1, 2]],[[0, 1], [-1, -2]])

    # Default argument in function
    x, y = T.dscalars('x', 'y')
    z = x + y
    f1 = theano.function([x, theano.In(y, value=1)], z)
    print f1(1)

    # shared variables
    state = theano.shared(0)
    state1 = theano.shared(1)
    inc = T.iscalar('inc')
    accumulator = theano.function([inc], updates={state: state+inc, state1: state1+inc})
    accumulator(1)
    print state.get_value()
    accumulator(2)
    print state.get_value()

    # givens
    fn_of_state = state*2 + inc
    foo = T.scalar(dtype=state.dtype)
    skip_shared = theano.function([inc, foo], fn_of_state, givens={state: foo})
    print skip_shared(1, 3)
    print state.get_value()

    # random numbers
    srng = T.shared_randomstreams.RandomStreams(seed=2345)


if __name__ == "__main__":
    main()