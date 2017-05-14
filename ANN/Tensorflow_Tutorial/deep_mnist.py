
import sys
import os.path
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import pylab

__mnist__ = input_data.read_data_sets("MNIST_data/", one_hot=True)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bais_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                         strides=[1, 2, 2, 1], padding='SAME')

def main():
    """ Main """
    one_conv_deep_mnist()
    #simple_mnist()

def one_conv_deep_mnist():
    x = tf.placeholder(dtype=tf.float32, shape=[None, 784])
    _y = tf.placeholder(dtype=tf.float32, shape=[None, 10])

    # 1. First convolution layer
    W_conv1 = weight_variable([5,5,1,32])
    b_conv1 = bais_variable([32])
    
    x_img = tf.reshape(x, [-1, 28, 28, 1])

    h_conv1 = tf.nn.relu(conv2d(x_img, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # 2. Fully connected layer
    W_fc1 = weight_variable([14*14*32, 1024])
    b_fc1 = bais_variable([1024])

    # Reshape 2d image to flat for input in fully connected layer
    h_pool1_flat = tf.reshape(h_pool1, [-1, 14*14*32])
    
    # output of fully connected layer
    h_fc1 = tf.nn.relu(tf.matmul(h_pool1_flat, W_fc1) + b_fc1)

    # 3. Softmax layer
    W_softmax = weight_variable([1024, 10])
    b_softmax = bais_variable([10])

    y_conv = tf.nn.relu(tf.matmul(h_fc1, W_softmax) + b_softmax)

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=_y, logits=y_conv))
    train_fn = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    
    result = tf.equal(tf.argmax(_y, axis=1), tf.argmax(y_conv, axis=1))
    calculate_accuracy = tf.reduce_mean(tf.cast(result, tf.float32))

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    saver = tf.train.Saver()
    file_path = "deep_model/test.model"

    if os.path.isfile(file_path+".meta"):
        # Restore model
        saver.restore(sess, file_path)
    else:
        costs = []
        for i in range(10000):
            batch = __mnist__.train.next_batch(100)
            
            if i%100 == 0:
                train_accuracy = sess.run(calculate_accuracy, feed_dict={x: batch[0], _y: batch[1]})
                print "\nTrain accuracy: {0}".format(train_accuracy)
            
            _, temp_cost = sess.run([train_fn, cross_entropy], feed_dict={x: batch[0], _y: batch[1]})
            costs.append(temp_cost)
            sys.stderr.write('\rEpoch: %d/%d' % (i+1, 7000))
            sys.stderr.flush()
        
        # Save model
        save_path = saver.save(sess, file_path)
        print "\nSaved at {0}".format(save_path)
    
    #x1 = __mnist__.train.images[0].reshape(1,784)
    #y1 = __mnist__.train.labels[0].reshape(1,10)

    #test_accuracy = sess.run(calculate_accuracy, feed_dict={x: x1, _y: y1})
    test_accuracy = sess.run(calculate_accuracy, feed_dict={x: __mnist__.test.images, _y: __mnist__.test.labels})

    print "\nTest accuracy: {0}".format(test_accuracy)
   
    #print "\nLabel: {0}".format(np.argmax(y1)) 
    #pylab.imshow(x1.reshape(28,28))
    #pylab.show()

def simple_mnist():
    x = tf.placeholder(dtype=tf.float32, shape=[None, 784])
    y_ = tf.placeholder(dtype=tf.float32, shape=[None, 10])
    
    w = weight_variable([784,10])
    b = bais_variable([10])
    y = tf.matmul(x, w) + b
    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_fn = tf.train.GradientDescentOptimizer(0.5).minimize(cost)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # Add an op to save and restore model
    saver = tf.train.Saver()
    file_path = "tmp/test.model"

    if os.path.isfile(file_path+".meta"):
        saver.restore(sess, file_path)
    else:
        costs = []
        for _ in range(2000):
            batch_xs, batch_ys = __mnist__.train.next_batch(100)
            _, temp_cost = sess.run([train_fn, cost], feed_dict={x: batch_xs, y_: batch_ys})
            costs.append(temp_cost)
        
        save_path = saver.save(sess, file_path)
        print "Saved at {0}".format(save_path)

    result = tf.equal(tf.argmax(y, axis=1), tf.argmax(y_, axis=1))
    calculate_accuracy = tf.reduce_mean(tf.cast(result, tf.float32))

    accuracy = sess.run(calculate_accuracy, feed_dict={x: __mnist__.train.images, y_: __mnist__.train.labels})
    print accuracy

    #plt.plot(costs, '-')
    #plt.show()

if __name__ == "__main__":
    main()