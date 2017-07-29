# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 00:23:58 2017

@author: Aniket Pant
"""

import tensorflow as tf

# creating tensorflow constant
constant = tf.constant(20, name='constant')

# create tensorflow variables
b = tf.variable(2.0, name='b')
c = tf.variable(1.0, name='c')

# neural network operations to be passed through net
d = tf.add(b ,c, name='d')
e = tf.add(c, constant, name='e')
a = tf.multiple(d, e, name='a')

# generating instance (model, like scikit)
init = tf.global_variables_initializer()

# Starting TFlow session
with tf.Session() as sess:
    sess.run(init)
    a_out = sess.run(a)
    print("Variables a is {}".format(a_out))