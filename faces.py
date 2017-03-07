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
import tensorflow as tf

#---------------------------------- Running The Code ----------------------------------#
run_part7 = False # Training the network
run_part8 = False # Training and experimenting with the outliers
run_part9 = False # Reading from the file and visualizing 2 actors
#--------------------------------------------------------------------------------------#

def load_data(training_size):
    '''
    Loads data from /data directory and creates a dictionary M with keys [train0, test0, validation0, train1 ...]
    Such that each index represents data for a separate class (total 6 classes)
    :return: a dictionary specified above
    '''
    M = {}
    actors = ['carell','hader','baldwin', 'drescher', 'ferrera', 'chenoweth']
    for i in range(6):
        x_train = np.zeros((training_size, 1024))
        x_test = np.zeros((30, 1024))
        x_validation = np.zeros((15, 1024))
        j = 0
        for filename in os.listdir("data/"+actors[i]+"/training/"):
            if j >= training_size: break
            # Filter some system files that may appear in a folder:
            if filename.endswith(".png"):
                im = imread("data/" + actors[i] + "/training/" + filename)
                im = (im - 127.) / 255
                x_train[j, :] = im.flatten().T
                j += 1
        j = 0
        for filename in os.listdir("data/" + actors[i] + "/test/"):
            if j >= 30: break
            # Filter some system files that may appear in a folder:
            if filename.endswith(".png"):
                im = imread("data/" + actors[i] + "/test/" + filename)
                im = (im - 127.) / 255
                x_test[j, :] = im.flatten().T
                j += 1
        j = 0
        for filename in os.listdir("data/" + actors[i] + "/validation/"):
            if j >= 15: break
            # Filter some system files that may appear in a folder:
            if filename.endswith(".png"):
                im = imread("data/" + actors[i] + "/validation/" + filename)
                im = (im - 127.) / 255
                x_validation[j, :] = im.flatten().T
                j += 1

        M['train' + str(i)] = x_train
        M['test' + str(i)] = x_test
        M['validation' + str(i)] = x_validation

    return M

def load_outliers_data(training_size=40):
    '''
    Loads data from /data_outliers directory and creates a dictionary M with keys [train0, test0, validation0, train1 ...]
    Such that each index represents data for a separate class (total 6 classes)
    :return: a dictionary specified above
    '''
    M = {}
    actors = ['carell','hader','baldwin', 'drescher', 'ferrera', 'chenoweth']
    for i in range(6):
        x_train = np.zeros((training_size, 1024))
        x_test = np.zeros((30, 1024))
        x_validation = np.zeros((15, 1024))
        j = 0
        for filename in os.listdir("data_outliers/"+actors[i]+"/training/"):
            if j >= training_size: break
            # Filter some system files that may appear in a folder:
            if filename.endswith(".png"):
                im = imread("data_outliers/" + actors[i] + "/training/" + filename)
                im = (im - 127.) / 255
                x_train[j, :] = im.flatten().T
                j += 1
        j = 0
        for filename in os.listdir("data_outliers/" + actors[i] + "/test/"):
            if j >= 30: break
            # Filter some system files that may appear in a folder:
            if filename.endswith(".png"):
                im = imread("data_outliers/" + actors[i] + "/test/" + filename)
                im = (im - 127.) / 255
                x_test[j, :] = im.flatten().T
                j += 1
        j = 0
        for filename in os.listdir("data_outliers/" + actors[i] + "/validation/"):
            if j >= 15: break
            # Filter some system files that may appear in a folder:
            if filename.endswith(".png"):
                im = imread("data_outliers/" + actors[i] + "/validation/" + filename)
                im = (im - 127.) / 255
                x_validation[j, :] = im.flatten().T
                j += 1

        M['train' + str(i)] = x_train
        M['test' + str(i)] = x_test
        M['validation' + str(i)] = x_validation

    return M

def get_train_batch(M, N, seed):
    '''
    Parses the data dictionary and returns a subset of training set specified by the batch size N
    :param M: dictionary with all the data
    :param N: size of a batch
    :return: x and y matrices containing the training data and the labels in the correct format
    '''
    n = N/6
    batch_xs = zeros((0, 1024))
    batch_y_s = zeros( (0, 6))
    
    train_k =  ["train"+str(i) for i in range(6)]
    train_size = len(M[train_k[0]])
    
    for k in range(6):
        random.seed(seed + k)
        train_size = len(M[train_k[k]])
        idx = array(random.permutation(train_size)[:n])
        batch_xs = vstack((batch_xs, ((array(M[train_k[k]])[idx]))  ))
        one_hot = zeros(6)
        one_hot[k] = 1
        batch_y_s = vstack((batch_y_s,   tile(one_hot, (n, 1))   ))
    return batch_xs, batch_y_s

def get_validation(M):
    '''
    Parses the data dictionary and returns the entire validation set
    :param M: dictionary with all the data
    :return: x and y matrices containing the validation data and the labels in the correct format
    '''
    batch_xs = zeros((0, 1024))
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
    batch_xs = zeros((0, 1024))
    batch_y_s = zeros( (0, 6))
    
    test_k =  ["test"+str(i) for i in range(6)]
    for k in range(6):
        batch_xs = vstack((batch_xs, ((array(M[test_k[k]])[:]))  ))

        one_hot = zeros(6)
        one_hot[k] = 1
        batch_y_s = vstack((batch_y_s,   tile(one_hot, (len(M[test_k[k]]), 1))   ))
    return batch_xs, batch_y_s

def get_test_subset(M, actor_indices):
    '''
    Parses the data dictionary and returns the entire test set for specific actors
    :param M: dictionary with all the data
    :param actor_indices: indices of particular actors we are interested in
    :return: x and y matrices containing the test data and the labels in the correct format
    '''
    batch_xs = zeros((0, 1024))
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
    batch_xs = zeros((0, 1024))
    batch_y_s = zeros( (0, 6))
    
    train_k =  ["train"+str(i) for i in range(6)]
    for k in range(6):
        batch_xs = vstack((batch_xs, ((array(M[train_k[k]])[:]))  ))
        one_hot = zeros(6)
        one_hot[k] = 1
        batch_y_s = vstack((batch_y_s,   tile(one_hot, (len(M[train_k[k]]), 1))   ))
    return batch_xs, batch_y_s


# ---------------------------------------------Definitions end------------------------------------------------------
if run_part7:
    # Initialize network paramters
    training_size = 15
    batch_size = 15  # be carefuly since it cant exceed training_size*6
    nhid = 100
    lam = 0.0001
    read_from_file = False
    total_iterations = 3000

    # Initialize Tensor Flow variables
    M = load_data(training_size)
    x = tf.placeholder(tf.float32, [None, 1024])

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
        W0 = tf.Variable(tf.random_normal([1024, nhid], stddev=0.01))
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
    plt.savefig("experiments_part7/"+"train_size"+str(training_size)+"batches"+str(batch_size)+"lam"+str(lam)+"nhin"+str(nhid)+".png")

    cPickle.dump(snapshot, open("experiments_part7/"+"train_size"+str(training_size)+"batches"+str(batch_size)+"lam"+str(lam)+"nhin"+str(nhid)+".pkl", "w"))
    result_record = open("experiments_part7/results.txt", 'a')
    result_record.write('\n=======\nUsing '+str(nhid)+' hidden units\n')
    result_record.write(str(batch_size)+' batch size\n')
    result_record.write(str(lam)+' lambda\n')
    result_record.write(str(training_size)+' Training size')
    result_record.write('\nFinal test accuracy: '+ str(results[-1][1]))
    result_record.write('\nFinal train accuracy: '+ str(results[-1][0]))
    result_record.write('\nFinal validation accuracy: '+ str(results[-1][2]))
    result_record.close()

if run_part8:
    for lam in [0,0.0001,0.0003,0.001,0.003,0.01,0.03,0.1,0.3,1]:
        # Initialize network paramters
        training_size = 40
        batch_size = 30  # be carefuly since it cant exceed training_size*6
        nhid = 500
        read_from_file = False
        total_iterations = 3000

        # Initialize Tensor Flow variables
        M = load_outliers_data(training_size)
        x = tf.placeholder(tf.float32, [None, 1024])

        W0 = tf.Variable(tf.random_normal([1024, nhid], stddev=0.01))
        b0 = tf.Variable(tf.random_normal([nhid], stddev=0.01))

        W1 = tf.Variable(tf.random_normal([nhid, 6], stddev=0.01))
        b1 = tf.Variable(tf.random_normal([6], stddev=0.01))

        layer1 = tf.nn.tanh(tf.matmul(x, W0) + b0)
        layer2 = tf.matmul(layer1, W1) + b1

        y = tf.nn.softmax(layer2)
        y_ = tf.placeholder(tf.float32, [None, 6])

        decay_penalty = lam * tf.reduce_sum(tf.square(W0)) + lam * tf.reduce_sum(tf.square(W1))
        reg_NLL = -tf.reduce_sum(y_ * tf.log(y)) + decay_penalty

        train_step = tf.train.AdamOptimizer(0.0005).minimize(reg_NLL)

        init = tf.initialize_all_variables()
        sess = tf.Session()
        sess.run(init)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
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
                print "i = ", i

                test_accuracy = sess.run(accuracy, feed_dict={x: test_x, y_: test_y})
                validation_accuracy = sess.run(accuracy, feed_dict={x: val_x, y_: val_y})

                print "Test:", test_accuracy
                print "Validation:", validation_accuracy
                batch_xs, batch_ys = get_train(M)

                train_accuracy = sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys})
                print "Train:", train_accuracy
                print "Penalty:", sess.run(decay_penalty)
                results.append([train_accuracy, test_accuracy, validation_accuracy])


        result_record = open("experiments_part7/results.txt", 'a')
        result_record.write('\n=======\nUsing ' + str(nhid) + ' hidden units\n')
        result_record.write(str(batch_size) + ' batch size\n')
        result_record.write(str(lam) + ' lambda\n')
        result_record.write(str(training_size) + ' Training size')
        result_record.write('\nFinal test accuracy: ' + str(results[-1][1]))
        result_record.write('\nFinal train accuracy: ' + str(results[-1][0]))
        result_record.write('\nFinal validation accuracy: ' + str(results[-1][2]))
        result_record.close()

if run_part9:
    # Indices are: [carell','hader','baldwin', 'drescher', 'ferrera', 'chenoweth']
    actor_indices = [1, 3]

    # Initialize variables specified by the .pkl package
    snapshot = cPickle.load(open('best_FC_weights.pkl'))
    bias0 = snapshot["b0"]
    bias1 = snapshot["b1"]
    w0 =  snapshot["W0"]
    w1 = snapshot["W1"]

    M = load_data(90)
    test_x, test_y = get_test_subset(M, actor_indices)

    # 2 entries are a top neuron index and the ouput value for a particular actor
    top_n1 = [0,0]
    top_n2 = [0,0]

    # Iterate through all the neurons and find the most sensitive one
    # The most sensitive one is the one producing the heightest output for an actor
    for i in range(w0.shape[1]):
        output = zeros(6)
        for j in range(60):
            first_layer = tanh(dot(w0.T[i], test_x[j]) + bias0[i])
            #first_layer = 1./(1.+np.exp(dot(test_x[j], w0.T[i].T) + bias0[i]))
            output += dot(first_layer,w1[i].T) + bias1

        if top_n1[1] < output[actor_indices[0]]:
            top_n1[0] = i
            top_n1[1] = output[actor_indices[0]]

        if top_n2[1] < output[actor_indices[1]]:
            top_n2[0] = i
            top_n2[1] = output[actor_indices[1]]

    # Reshape and save weights for the most sensitive neuron as an image
    img1 = np.reshape(w0.T[top_n1[0]]+bias0[top_n1[0]],(32, 32))
    img2 = np.reshape(w0.T[top_n2[0]]+bias0[top_n2[0]], (32, 32))
    imsave('actor1.png', img1)
    imsave('actor2.png', img2)

    # Different visualization method
    testar = dot(w0, w1).T[actor_indices[0]]
    img = np.reshape(testar, (32, 32))
    imsave('actor1_allweights.png', img)

    testar = dot(w0, w1).T[actor_indices[1]]
    img = np.reshape(testar, (32, 32))
    imsave('actor2_allweights.png', img)