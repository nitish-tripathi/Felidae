
import pylab
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# read data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# show images
"""
img_ = mnist.train.images[1].reshape(28,28)
print np.argmax(mnist.train.labels[1])
pylab.imshow(img_)
pylab.show()
"""
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.matmul(x, W) + b

y_ = tf.placeholder(tf.float32, [None, 10])

#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

costs = []
for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    _, loss_val = sess.run([train_step, cross_entropy], feed_dict={x: batch_xs, y_: batch_ys})
    #print "Cost: {0}".format(loss_val)
    costs.append(loss_val)

correct_predictions = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

acc = sess.run(accuracy, feed_dict={x: mnist.train.images, y_: mnist.train.labels})
print acc
