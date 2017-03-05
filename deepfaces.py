################################################################################
#Michael Guerzhoy and Davi Frossard, 2016
#AlexNet implementation in TensorFlow, with weights
#Details: 
#http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
#
#With code from https://github.com/ethereon/caffe-tensorflow
#Model from  https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet
#Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow
#
#
################################################################################

from numpy import *
import os
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
from scipy.stats import norm
import cPickle
from scipy.io import loadmat
import pickle

import tensorflow as tf
from caffe_classes import class_names

#---------------------------------- Running The Code ----------------------------------#
run_part10 = False # Training the network
run_part11 = True # Reading from the file and visualizing 2 actors
#--------------------------------------------------------------------------------------#

# Get the AlexNet weights from bvlc_alexnet.npy
NET_DATA = load(open("bvlc_alexnet.npy", "rb"), encoding="latin1").item()

def load_data(training_size):
    '''
    Loads data from /data directory and creates a dictionary M with keys [train0, test0, validation0, train1 ...]
    Such that each index represents data for a separate class (total 6 classes)
    :return: a dictionary specified above
    '''
    M = {}
    actors = ['carell', 'hader', 'baldwin', 'drescher', 'ferrera', 'chenoweth']
    for i in range(6):
        x_train = np.zeros((training_size, 227, 227, 3))
        x_test = np.zeros((30, 227, 227, 3))
        x_validation = np.zeros((15, 227, 227, 3))
        j = 0
        for filename in os.listdir("data_color_227/" + actors[i] + "/training/"):
            if j >= training_size: break
            # Filter some system files that may appear in a folder:
            if filename.endswith(".png"):
                im = imread("data_color_227/" + actors[i] + "/training/" + filename).astype(float32)
                im = (im - 127.) / 255
                if im.shape == (227,227,4):
                    im = im[:,:,:3]
                x_train[j, :, :, :] = im
                j += 1
        j = 0
        for filename in os.listdir("data_color_227/" + actors[i] + "/test/"):
            if j >= 30: break
            # Filter some system files that may appear in a folder:
            if filename.endswith(".png"):
                im = imread("data_color_227/" + actors[i] + "/test/" + filename).astype(float32)
                im = (im - 127.) / 255
                if im.shape == (227,227,4):
                    im = im[:,:,:3]
                x_test[j, :, :, :] = im
                j += 1
        j = 0
        for filename in os.listdir("data_color_227/" + actors[i] + "/validation/"):
            if j >= 15: break
            # Filter some system files that may appear in a folder:
            if filename.endswith(".png"):
                im = imread("data_color_227/" + actors[i] + "/validation/" + filename).astype(float32)
                im = (im - 127.) / 255
                if im.shape == (227,227,4):
                    im = im[:,:,:3]
                x_validation[j, :, :, :] = im
                j += 1

        M['train' + str(i)] = x_train
        M['test' + str(i)] = x_test
        M['validation' + str(i)] = x_validation


    return extract_feature_vectors(M)

def get_train_batch(M, N, seed):
    '''
    Parses the data dictionary and returns a subset of training set specified by the batch size N
    :param M: dictionary with all the data
    :param N: size of a batch
    :return: x and y matrices containing the training data and the labels in the correct format
    '''
    n = N / 6
    batch_xs = zeros((0, 13*13*384))
    batch_y_s = zeros((0, 6))

    train_k = ["train" + str(i) for i in range(6)]
    train_size = len(M[train_k[0]])

    for k in range(6):
        random.seed(seed + k)
        train_size = len(M[train_k[k]])
        idx = array(random.permutation(train_size)[:n])
        batch_xs = vstack((batch_xs, ((array(M[train_k[k]])[idx]))))
        one_hot = zeros(6)
        one_hot[k] = 1
        batch_y_s = vstack((batch_y_s, tile(one_hot, (n, 1))))
    return batch_xs, batch_y_s

def get_validation(M):
    '''
    Parses the data dictionary and returns the entire validation set
    :param M: dictionary with all the data
    :return: x and y matrices containing the validation data and the labels in the correct format
    '''
    batch_xs = zeros((0, 13*13*384))
    batch_y_s = zeros((0, 6))

    validation_k = ["validation" + str(i) for i in range(6)]
    for k in range(6):
        batch_xs = vstack((batch_xs, ((array(M[validation_k[k]])[:]))))
        one_hot = zeros(6)
        one_hot[k] = 1
        batch_y_s = vstack((batch_y_s, tile(one_hot, (len(M[validation_k[k]]), 1))))
    return batch_xs, batch_y_s

def get_test(M):
    '''
    Parses the data dictionary and returns the entire test set
    :param M: dictionary with all the data
    :return: x and y matrices containing the test data and the labels in the correct format
    '''
    batch_xs = zeros((0, 13*13*384))
    batch_y_s = zeros((0, 6))

    test_k = ["test" + str(i) for i in range(6)]
    for k in range(6):
        batch_xs = vstack((batch_xs, ((array(M[test_k[k]])[:]))))

        one_hot = zeros(6)
        one_hot[k] = 1
        batch_y_s = vstack((batch_y_s, tile(one_hot, (len(M[test_k[k]]), 1))))
    return batch_xs, batch_y_s

def get_test_subset(M, actor_indices):
    '''
    Parses the data dictionary and returns the entire test set for specific actors
    :param M: dictionary with all the data
    :param actor_indices: indices of particular actors we are interested in
    :return: x and y matrices containing the test data and the labels in the correct format
    '''
    batch_xs = zeros((0, 13*13*384))
    batch_y_s = zeros((0, 6))

    test_k = ["test" + str(i) for i in actor_indices]
    for k in range(len(actor_indices)):
        batch_xs = vstack((batch_xs, ((array(M[test_k[k]])[:]))))

        one_hot = zeros(6)
        one_hot[k] = 1
        batch_y_s = vstack((batch_y_s, tile(one_hot, (len(M[test_k[k]]), 1))))
    return batch_xs, batch_y_s

def get_train(M):
    '''
    Parses the data dictionary and returns the entire training set
    :param M: dictionary with all the data
    :return: x and y matrices containing the training data and the labels in the correct format
    '''
    batch_xs = zeros((0, 13*13*384))
    batch_y_s = zeros((0, 6))

    train_k = ["train" + str(i) for i in range(6)]
    for k in range(6):
        batch_xs = vstack((batch_xs, ((array(M[train_k[k]])[:]))))
        one_hot = zeros(6)
        one_hot[k] = 1
        batch_y_s = vstack((batch_y_s, tile(one_hot, (len(M[train_k[k]]), 1))))
    return batch_xs, batch_y_s


def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w,  padding="VALID", group=1):
    '''From https://github.com/ethereon/caffe-tensorflow
    '''
    c_i = input.get_shape()[-1]
    assert c_i%group==0
    assert c_o%group==0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
    
    
    if group==1:
        conv = convolve(input, kernel)
    else:
        input_groups =  tf.split(input, group, 3)   #tf.split(3, group, input)
        kernel_groups = tf.split(kernel, group, 3)  #tf.split(3, group, kernel) 
        output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
        conv = tf.concat(output_groups, 3)          #tf.concat(3, output_groups)
    return  tf.reshape(tf.nn.bias_add(conv, biases), [-1]+conv.get_shape().as_list()[1:])

def extract_feature_vectors(data_dict):
    train_x = zeros((1, 227, 227, 3)).astype(float32)
    xdim = train_x.shape[1:]

    new_dict = dict.fromkeys(data_dict.keys())

    x = tf.placeholder(tf.float32, (None,) + xdim)

    #conv1
    #conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
    k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4
    conv1W = tf.Variable(NET_DATA["conv1"][0])
    conv1b = tf.Variable(NET_DATA["conv1"][1])
    conv1_in = conv(x, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
    conv1 = tf.nn.relu(conv1_in)

    #lrn1
    #lrn(2, 2e-05, 0.75, name='norm1')
    radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
    lrn1 = tf.nn.local_response_normalization(conv1,
                                                      depth_radius=radius,
                                                      alpha=alpha,
                                                      beta=beta,
                                                      bias=bias)

    #maxpool1
    #max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
    k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
    maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)


    #conv2
    #conv(5, 5, 256, 1, 1, group=2, name='conv2')
    k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2
    conv2W = tf.Variable(NET_DATA["conv2"][0])
    conv2b = tf.Variable(NET_DATA["conv2"][1])
    conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv2 = tf.nn.relu(conv2_in)


    #lrn2
    #lrn(2, 2e-05, 0.75, name='norm2')
    radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
    lrn2 = tf.nn.local_response_normalization(conv2,
                                                      depth_radius=radius,
                                                      alpha=alpha,
                                                      beta=beta,
                                                      bias=bias)

    #maxpool2
    #max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
    k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
    maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

    #conv3
    #conv(3, 3, 384, 1, 1, name='conv3')
    k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
    conv3W = tf.Variable(NET_DATA["conv3"][0])
    conv3b = tf.Variable(NET_DATA["conv3"][1])
    conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv3 = tf.nn.relu(conv3_in)

    #conv4
    #conv(3, 3, 384, 1, 1, group=2, name='conv4')
    k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
    conv4W = tf.Variable(NET_DATA["conv4"][0])
    conv4b = tf.Variable(NET_DATA["conv4"][1])
    conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv4 = tf.nn.relu(conv4_in)

    #
    # #conv5
    # #conv(3, 3, 256, 1, 1, group=2, name='conv5')
    # k_h = 3; k_w = 3; c_o = 256; s_h = 1; s_w = 1; group = 2
    # conv5W = tf.Variable(NET_DATA["conv5"][0])
    # conv5b = tf.Variable(NET_DATA["conv5"][1])
    # conv5_in = conv(conv4, conv5W, conv5b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    # conv5 = tf.nn.relu(conv5_in)
    #
    # #maxpool5
    # #max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
    # k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
    # maxpool5 = tf.nn.max_pool(conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)
    #
    # #fc6
    # #fc(4096, name='fc6')
    # fc6W = tf.Variable(NET_DATA["fc6"][0])
    # fc6b = tf.Variable(NET_DATA["fc6"][1])
    # fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [-1, int(prod(maxpool5.get_shape()[1:]))]), fc6W, fc6b)
    #
    # #fc7
    # #fc(4096, name='fc7')
    # fc7W = tf.Variable(NET_DATA["fc7"][0])
    # fc7b = tf.Variable(NET_DATA["fc7"][1])
    # fc7 = tf.nn.relu_layer(fc6, fc7W, fc7b)
    #
    # #fc8
    # #fc(1000, relu=False, name='fc8')
    # fc8W = tf.Variable(NET_DATA["fc8"][0])
    # fc8b = tf.Variable(NET_DATA["fc8"][1])
    # fc8 = tf.nn.xw_plus_b(fc7, fc8W, fc8b)
    #
    #
    # #prob
    # #softmax(name='prob'))
    # prob = tf.nn.softmax(fc8)

    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    for key in new_dict.keys():
        new_dict[key] = zeros((data_dict[key].shape[0], 13*13*384))
        for i in range(data_dict[key].shape[0]):
            image = data_dict[key][i]
            image[:, :, 0], image[:, :, 2] = image[:, :, 2], image[:, :, 0]
            output = sess.run(conv4, feed_dict={x: [image]})
            new_dict[key][i, :] = output.flatten().T
    print('Feature vectors extracted on all data')
    return new_dict

# ---------------------------------------------Definitions end------------------------------------------------------
if run_part10:
    # Initialize network paramters
    training_size = 90
    batch_size = 30  # be careful since it cant exceed training_size*6
    nhid = 50
    lam = 0.0001
    read_from_file = False
    total_iterations = 1000

    # Initialize Tensor Flow variables
    M = load_data(training_size)
    x = tf.placeholder(tf.float32, [None, 13*13*384])

    # If reading from file, set up the best configuration for TF
    if read_from_file:
        snapshot = cPickle.load(open('best_FC_weights.pkl'))
        W0 = tf.Variable(snapshot["W0"])
        b0 = tf.Variable(snapshot["b0"])
        W1 = tf.Variable(snapshot["W1"])
        b1 = tf.Variable(snapshot["b1"])
        training_size = 90
        batch_size = 180
        nhid = 30
        lam = 0.0001
        total_iterations = 1
    else:
        W0 = tf.Variable(tf.random_normal([13*13*384, nhid], stddev=0.01))
        b0 = tf.Variable(tf.random_normal([nhid], stddev=0.01))

        W1 = tf.Variable(tf.random_normal([nhid, 6], stddev=0.01))
        b1 = tf.Variable(tf.random_normal([6], stddev=0.01))


    layer1 = tf.nn.tanh(tf.matmul(x, W0)+b0)
    layer2 = tf.matmul(layer1, W1)+b1

    y = tf.nn.softmax(layer2)
    y_ = tf.placeholder(tf.float32, [None, 6])

    decay_penalty =lam*tf.reduce_sum(tf.square(W0))+lam*tf.reduce_sum(tf.square(W1))
    reg_NLL = -tf.reduce_sum(y_*tf.log(y))+decay_penalty

    train_step = tf.train.AdamOptimizer(0.0005).minimize(reg_NLL)

    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    test_x, test_y = get_test(M)
    val_x, val_y = get_validation(M)

    # Run the TF, collect accuracies in a separate array
    results = []
    for i in range(total_iterations):
      seed = i
      batch_xs, batch_ys = get_train_batch(M, batch_size, seed)
      sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

      if i % 10 == 0:
        print "i = ",i

        test_accuracy = sess.run(accuracy, feed_dict={x: test_x, y_: test_y})
        validation_accuracy = sess.run(accuracy, feed_dict={x: val_x, y_: val_y})

        print "Test:", test_accuracy
        print "Validation:", validation_accuracy
        batch_xs, batch_ys = get_train(M)

        train_accuracy = sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys})
        print "Train:", train_accuracy
        print "Penalty:", sess.run(decay_penalty)
        results.append([train_accuracy, test_accuracy, validation_accuracy])

    if read_from_file: exit()
    # Save weights for later use
    snapshot = {}
    snapshot["W0"] = sess.run(W0)
    snapshot["W1"] = sess.run(W1)
    snapshot["b0"] = sess.run(b0)
    snapshot["b1"] = sess.run(b1)

    # Plot the learning curve
    x_axis = linspace(0, total_iterations, len(results))
    training_results = [results[i][0] for i in range(len(results))]
    test_results = [results[i][1] for i in range(len(results))]
    validation_results = [results[i][2] for i in range(len(results))]

    plt_training = plt.plot(x_axis, training_results, label='Training')
    plt_test = plt.plot(x_axis, test_results, label='Test')
    plt_validation = plt.plot(x_axis, validation_results, label='Validation')
    plt.ylim([0.2, 1.05])
    plt.xlabel('# of Iterations')
    plt.ylabel('Performance')
    plt.title('Performance vs. # of iterations')
    plt.legend(["Training", "Test", "Validation"], loc=7)
    plt.savefig("deepnet_curve.png")

    cPickle.dump(snapshot, open("deepnet_weights.pkl", "w"))

if run_part11:
    # Indices are: [carell','hader','baldwin', 'drescher', 'ferrera', 'chenoweth']
    actor_indices = [1, 3]

    # Initialize variables specified by the .pkl package
    snapshot = cPickle.load(open('deepnet_weights.pkl'))
    bias0 = snapshot["b0"]
    bias1 = snapshot["b1"]
    w0 =  snapshot["W0"]
    w1 = snapshot["W1"]

    M = load_data(90)
    test_x, test_y = get_test_subset(M, actor_indices)

    # 2 entries are a top neuron index and the ouput value for a particular actor
    top_n1 = [0, 0]
    top_n2 = [0, 0]

    # Iterate through all the neurons and find the most sensitive one
    # The most sensitive one is the one producing the heightest output for an actor
    for i in range(w0.shape[1]):
        output = zeros(6)
        for j in range(60):
            first_layer = tanh(dot(w0.T[i], test_x[j]) + bias0[i])
            # first_layer = 1./(1.+np.exp(dot(test_x[j], w0.T[i].T) + bias0[i]))
            output += dot(first_layer, w1[i].T) + bias1

        if top_n1[1] < output[actor_indices[0]]:
            top_n1[0] = i
            top_n1[1] = output[actor_indices[0]]

        if top_n2[1] < output[actor_indices[1]]:
            top_n2[0] = i
            top_n2[1] = output[actor_indices[1]]

    # Reshape and save weights for the most sensitive neuron as an image
    img1 = np.reshape(w0.T[top_n1[0]] + bias0[top_n1[0]], (8*13*2, 8*13*3))
    img2 = np.reshape(w0.T[top_n2[0]] + bias0[top_n2[0]], (8*13*2, 8*13*3))
    imsave('actor1_deep.png', img1)
    imsave('actor2_deep.png', img2)

    # Different visualization method
    testar = dot(w0, w1).T[actor_indices[0]]
    img = np.reshape(testar, (8*13*2, 8*13*3))
    imsave('actor1_allweights_deep.png', img)

    testar = dot(w0, w1).T[actor_indices[1]]
    img = np.reshape(testar, (8*13*2, 8*13*3))
    imsave('actor2_allweights_deep.png', img)
