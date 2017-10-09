import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataframe = pd.read_csv("tensorflow_first.csv")
dataframe = dataframe.drop(["index", "price", "sq_price"],axis=1)

dataframe = dataframe[0:10]

dataframe.loc[:,("y1")]=[1,1,1,0,0,1,0,1,1,1]
dataframe.loc[:, ("y2")] = dataframe["y1"] == 0
dataframe.loc[:, ("y2")] = dataframe["y2"].astype(int)
print dataframe

#preparing data for tensorflow

#convert features to tensors

inputX = dataframe.loc[:,["area","bathrooms"]].as_matrix()
inputY = dataframe.loc[:,["y1","y2"]].as_matrix()

print inputX , inputY

#write our hyperparameters

learning_rate = 0.000001
training_epochs = 10000
display_step = 50
n_samples = inputY.size

#computation graph

x = tf.placeholder(tf.float32,[None,2])
w = tf.Variable(tf.zeros([2,2]))

b = tf.Variable(tf.zeros([2]))

y_values = tf.add(tf.matmul(x,w),b)

y = tf.nn.softmax(y_values)

y_ = tf.placeholder(tf.float32, [None,2])

cost = tf.reduce_sum(tf.pow(y_ - y, 2)/(2*n_samples))

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for i in range(training_epochs):
    sess.run(optimizer, feed_dict={x: inputX, y_: inputY})

    if (i) % display_step == 0:
        cc = sess.run(cost, feed_dict={x: inputX, y_: inputY})
        print "Training step:", '%04d' % (i), "cost=", "{:.9f}".format(cc)

print "Optimization Finished!"
training_cost = sess.run(cost, feed_dict={x: inputX, y_: inputY})
print "Training cost=", training_cost, "W=", sess.run(w), "b=", sess.run(b), '\n'

print "\n"

print sess.run(y, feed_dict={x: inputX })
print sess.run(tf.nn.softmax([1., 2.]))


