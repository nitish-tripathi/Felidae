
import numpy as np
import theano
import theano.tensor as T

rng = np.random

N = 400
feats = 784

# generate dataset: D = (input_values, target_class)
D = (rng.randn(N, feats), rng.randint(size=N, low=0, high=2))
training_steps = 10000

# declare theano shared variables
x = T.dmatrix('x')
y = T.dvector('y')

# weights and baises
w = theano.shared(rng.randn(feats), name='w')
b = theano.shared(0., name='b')

print "Initial model:"
print w.shape.eval()
print D[0].shape
print D[1].shape
print b.get_value()

sigmoid = 1 / (1 + T.exp(-T.dot(x, w) -b))
prediction = sigmoid > 0.5
xent = -y*T.log(sigmoid) -(1-y)*T.log(1-sigmoid)
cost = xent.mean() + 0.01 * (w ** 2).sum()

gw, gb = T.grad(cost, [w, b])

train = theano.function(inputs=[x,y],
                        outputs=[prediction, xent],
                        updates={w: w-0.1*gw, b: b-0.1*gb})

predict = theano.function(inputs=[x], outputs=prediction)

for i in range(training_steps):
    pred, err = train(D[0], D[1])
    print np.mean(err)
