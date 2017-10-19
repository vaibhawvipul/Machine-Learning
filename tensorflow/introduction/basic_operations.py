__author__ = "vaibhawvipul"
import tensorflow as tf

#Hello World
hello_world = tf.constant("\nHello World!")

with tf.Session() as sess:
    print sess.run(hello_world)

#constant is used to define constants
a = tf.constant(2)
b = tf.constant(3)

with tf.Session() as sess: # creating tensorflow Session
    print "\na: %i" % sess.run(a), "b: %i" % sess.run(b)
    print "Multiplying a and b: %i" % sess.run(a*b)
    print "Adding a and b: %i" % sess.run(a+b)

#playing with variables
c = tf.placeholder(tf.int32)
d = tf.placeholder(tf.int32)

#we define some operations
add = tf.add(c,d)
mul = tf.multiply(c,d)

with tf.Session() as sess:
    print sess.run(mul, feed_dict={c:2,d:3})
    print sess.run(add, feed_dict={c:2,d:3})

#Defining matrix

#1X2 matrix
matrix1 = tf.constant([[3,3]])
#2X1 matrix
matrix2 = tf.constant([[2],[2]])

matrix_product = tf.matmul(matrix1,matrix2)

with tf.Session() as sess:
    print sess.run(matrix1)
    print sess.run(matrix2)
    print sess.run(matrix_product)
