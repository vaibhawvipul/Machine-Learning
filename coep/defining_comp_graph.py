import tensorflow as tf 

a = tf.constant(10)
b = tf.constant(5)

temp1 = tf.multiply(a,b)
temp2 = tf.add(a,b)
res = tf.divide(temp1,temp2)

tf.Print(res)