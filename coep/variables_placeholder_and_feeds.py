import tensorflow as tf 

a = tf.constant(3.0)
b = tf.placeholder(tf.float32,[])

c = a+b

with tf.Session() as session:
    print(session.run(c,feed_dict={a:3.0}))
    print(session.run(c,feed_dict={a:5.0}))


