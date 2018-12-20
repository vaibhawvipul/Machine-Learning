import tensorflow as tf 

a = tf.constant(3)
b = tf.constant(2)
c = tf.constant(10)

d = tf.divide(c,tf.add(a,b))

with tf.Session() as sess:
    print(sess.run(d))