__author__ = "vaibhawvipul"

import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data",one_hot=True)

# In this example, we limit mnist data
X_train, Y_train = mnist.train.next_batch(5000) #5000 for training (nn candidates)
X_test, Y_test = mnist.test.next_batch(200) #200 for testing

# tf Graph Input
x_train = tf.placeholder("float", [None, 784])
x_test = tf.placeholder("float", [784])

# Nearest Neighbor calculation using L1 Distance
# Calculate L1 Distance
distance = tf.reduce_sum(tf.abs(tf.add(x_train, tf.negative(x_test))), reduction_indices=1)

# Prediction: Get min distance index (Nearest neighbor)
pred = tf.arg_min(distance, 0)

accuracy = 0.

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:
    sess.run(init)

    for i in range(len(X_test)):
        # Get nearest neighbor
        nn_index = sess.run(pred, feed_dict={x_train: X_train, x_test: X_test[i, :]})

        # Get nearest neighbor class label and compare it to its true label
        print "Test", i, "Prediction:", np.argmax(Y_train[nn_index]), \
            "True Class:", np.argmax(Y_test[i])
        # Calculate accuracy

        if np.argmax(Y_train[nn_index]) == np.argmax(Y_test[i]):
            accuracy += 1./len(X_test)
    print "Done!"
    print "Accuracy:", accuracy
