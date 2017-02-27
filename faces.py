from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
from numpy import random

import cPickle

import os
from scipy.io import loadmat


t = int(time.time())
#t = 1454219613
print "t=", t
random.seed(t)


M = loadmat("mnist_all.mat")

import tensorflow as tf

def get_train_batch(M, N):
    n = N/10
    batch_xs = zeros((0, 28*28))
    batch_y_s = zeros( (0, 10))
    
    train_k =  ["train"+str(i) for i in range(10)]

    train_size = len(M[train_k[0]])
    #train_size = 5000
    
    for k in range(10):
        train_size = len(M[train_k[k]])
        idx = array(random.permutation(train_size)[:n])
        batch_xs = vstack((batch_xs, ((array(M[train_k[k]])[idx])/255.)  ))
        one_hot = zeros(10)
        one_hot[k] = 1
        batch_y_s = vstack((batch_y_s,   tile(one_hot, (n, 1))   ))
    return batch_xs, batch_y_s
    

def get_test(M):
    batch_xs = zeros((0, 28*28))
    batch_y_s = zeros( (0, 10))
    
    test_k =  ["test"+str(i) for i in range(10)]
    for k in range(10):
        batch_xs = vstack((batch_xs, ((array(M[test_k[k]])[:])/255.)  ))
        one_hot = zeros(10)
        one_hot[k] = 1
        batch_y_s = vstack((batch_y_s,   tile(one_hot, (len(M[test_k[k]]), 1))   ))
    return batch_xs, batch_y_s


def get_train(M):
    batch_xs = zeros((0, 28*28))
    batch_y_s = zeros( (0, 10))
    
    train_k =  ["train"+str(i) for i in range(10)]
    for k in range(10):
        batch_xs = vstack((batch_xs, ((array(M[train_k[k]])[:])/255.)  ))
        one_hot = zeros(10)
        one_hot[k] = 1
        batch_y_s = vstack((batch_y_s,   tile(one_hot, (len(M[train_k[k]]), 1))   ))
    return batch_xs, batch_y_s
        



x = tf.placeholder(tf.float32, [None, 784])

nhid = 300
W0 = tf.Variable(tf.random_normal([784, nhid], stddev=0.01))
b0 = tf.Variable(tf.random_normal([nhid], stddev=0.01))

W1 = tf.Variable(tf.random_normal([nhid, 10], stddev=0.01))
b1 = tf.Variable(tf.random_normal([10], stddev=0.01))

snapshot = cPickle.load(open("snapshot50.pkl"))
W0 = tf.Variable(snapshot["W0"])
b0 = tf.Variable(snapshot["b0"])
W1 = tf.Variable(snapshot["W1"])
b1 = tf.Variable(snapshot["b1"])


layer1 = tf.nn.tanh(tf.matmul(x, W0)+b0)
layer2 = tf.matmul(layer1, W1)+b1


y = tf.nn.softmax(layer2)
y_ = tf.placeholder(tf.float32, [None, 10])



lam = 0.00000
decay_penalty =lam*tf.reduce_sum(tf.square(W0))+lam*tf.reduce_sum(tf.square(W1))
reg_NLL = -tf.reduce_sum(y_*tf.log(y))+decay_penalty

train_step = tf.train.AdamOptimizer(0.0005).minimize(reg_NLL)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
test_x, test_y = get_test(M)


for i in range(5000):
  #print i  
  batch_xs, batch_ys = get_train_batch(M, 500)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
  
  
  if i % 1 == 0:
    print "i=",i
    print "Test:", sess.run(accuracy, feed_dict={x: test_x, y_: test_y})
    batch_xs, batch_ys = get_train(M)

    print "Train:", sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys})
    print "Penalty:", sess.run(decay_penalty)


    snapshot = {}
    snapshot["W0"] = sess.run(W0)
    snapshot["W1"] = sess.run(W1)
    snapshot["b0"] = sess.run(b0)
    snapshot["b1"] = sess.run(b1)
    cPickle.dump(snapshot,  open("new_snapshot"+str(i)+".pkl", "w"))