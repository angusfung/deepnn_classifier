from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy.ndimage import filters
from scipy.stats import norm
import urllib
from numpy import random
import cPickle
import os
from scipy.io import loadmat
import pickle

#---------------------------------- Running The Code ----------------------------------#
run_part1 = False  # generate 100 images
run_part3 = False  # finite differences
run_part4 = False  # plot the learning curve, print out the final performance results
run_part5 = True  # multinomial example
#--------------------------------------------------------------------------------------#

# Load the MNIST digit data
M = loadmat("mnist_all.mat")


#--------------------------------------- Part 1 -----------------------------------------#
def plot_figures(filename):
    '''
    Plots 10 samples for each digits from a set
    :param filename: filename e.g "train1", "train2"
    :return: None
    '''

    # generates 10 rand numbers
    random.seed(1)
    rand = random.random(10) * len(M[filename])

    # unflatten the images
    x1 = M[filename][int(rand[0])].reshape((28, 28))
    x2 = M[filename][int(rand[1])].reshape((28, 28))
    x3 = M[filename][int(rand[2])].reshape((28, 28))
    x4 = M[filename][int(rand[3])].reshape((28, 28))
    x5 = M[filename][int(rand[4])].reshape((28, 28))
    x6 = M[filename][int(rand[5])].reshape((28, 28))
    x7 = M[filename][int(rand[6])].reshape((28, 28))
    x8 = M[filename][int(rand[7])].reshape((28, 28))
    x9 = M[filename][int(rand[8])].reshape((28, 28))
    x10 = M[filename][int(rand[9])].reshape((28, 28))

    # Four axes, returned as a 2-d array
    f, axarr = plt.subplots(2, 5)
    axarr[0, 0].imshow(x1, cmap=cm.gray)
    axarr[0, 1].imshow(x2, cmap=cm.gray)
    axarr[0, 2].imshow(x3, cmap=cm.gray)
    axarr[0, 3].imshow(x4, cmap=cm.gray)
    axarr[0, 4].imshow(x5, cmap=cm.gray)
    axarr[1, 0].imshow(x6, cmap=cm.gray)
    axarr[1, 1].imshow(x7, cmap=cm.gray)
    axarr[1, 2].imshow(x8, cmap=cm.gray)
    axarr[1, 3].imshow(x9, cmap=cm.gray)
    axarr[1, 4].imshow(x10, cmap=cm.gray)

    # Fine-tune figure; hide x ticks for top plots and y ticks for right plots
    plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
    plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)

    plt.show()


#--------------------------------------- Part 2 -----------------------------------------#
def softmax(y):
    '''Returns the output of the softmax function for the matrix of output y. y
    is an NxM matrix where N is the number of outputs for a single case, and M
    is the number of cases'''

    return exp(y) / tile(sum(exp(y), 0), (len(y), 1))


def part2(x, weights):
    '''Returns a matrix of probabiltiies, the output from the cost function
       bias is the first row of the weight matrix (785 x 10)
    '''
    x = x / 255.0
    x = vstack((ones((1, x.shape[1])), x))  # include x_0
    y = dot(weights.T, x)
    return softmax(y)


#--------------------------------------- Part 3 -----------------------------------------#
def f(x, y, weights):
    '''@input prob is a NxM matrix of probabilities
       @input y is a NxM matrix where each column is the one-hot-representation
              of an image
              N is the number of outputs for a single case, M is the number of
              cases (e.g images)
       @input weights is a 785 x 10 matrix
              '''
    prob = part2(x, weights)
    return -sum(y * log(prob))

def df(x, y, weights):
    # Gradient function based on logistic regression
    prob = part2(x, weights)
    x = x / 255.0
    x = vstack((ones((1, x.shape[1])), x))
    return dot(x, (prob - y).T)

def part3():
    '''
    Compares finite difference approximation and vectorized gradient algorithm
    :return: None
    '''
    # initializing random variables to test
    m = 200

    random.seed(0)
    x = reshape(random.rand(784 * m), (784, m))
    random.seed(1)
    y = zeros((10, m))  # one-hot encoding matrix
    y[0, :] = 1
    random.seed(2)
    weights = reshape(random.rand(785 * 10), (785, 10))

    h = 0.00000001

    random.seed(4)
    dh = zeros((785, 10))
    dh[0, 0] = h

    print("Finite difference approximation at (1,1)")
    print (f(x, y, weights + dh) - f(x, y, weights - dh)) / (2 * h)
    print df(x, y, weights)

    dh[0, 0] = 0
    dh[0, 1] = h
    print("Finite difference approximation at (1,2)")
    print (f(x, y, weights + dh) - f(x, y, weights - dh)) / (2 * h)
    print df(x, y, weights)

    dh[0, 1] = 0
    dh[1, 1] = h
    print("Finite difference approximation at (2,2)")
    print (f(x, y, weights + dh) - f(x, y, weights - dh)) / (2 * h)
    print df(x, y, weights)

    dh[1, 1] = 0
    dh[2, 2] = h
    print("Finite difference approximation at (3,3)")
    print (f(x, y, weights + dh) - f(x, y, weights - dh)) / (2 * h)
    print df(x, y, weights)
    return

#--------------------------------------- Part 4 -----------------------------------------#
def save_weights(learning_weights, alpha):
    '''
    Saves the weights in a separate file for later use
    :param learning_weights: training weigts that will be saved
    :param learning_weights: alpha used in the training
    '''
    with open('learning_weights' + str(alpha) +'.pkl', 'wb') as output:
        pickle.dump(learning_weights, output, pickle.HIGHEST_PROTOCOL)

    return True

def reload_weights(alpha):
    '''
    Reloads the weights from a separate file
    :param learning_weights: training weigts that will be saved
    :param learning_weights: alpha used in the training
    :return: learning_weights (array of numpy arrays) loaded from the pkl file
    '''
    with open('learning_weights' + str(alpha) +'.pkl', 'rb') as input:
        learning_weights = pickle.load(input)
    return learning_weights

def grad_descent(f, df, x, y, init_t, alpha):
    '''
    Gradient descent optimizing algorithm
    :param f: cost function
    :param df: gradient
    :param x: x input matrix
    :param y: y input matrix
    :param init_t: initial vector theta
    :param alpha: step size
    :return: minimizing theta, value of the cost function
    Also modifies the learning weights global variable
    '''
    global learning_weights
    learning_weights = []
    EPS = 1e-10  # EPS = 10**(-5)
    prev_t = init_t - 10 * EPS
    t = init_t.copy()
    max_iter = 3000
    iter = 0
    while norm(t - prev_t) > EPS and iter < max_iter:
        prev_t = t.copy()
        t -= alpha * df(x, y, t)
        if iter % 100 == 0:
            curr_t = t.copy()
            learning_weights.append(curr_t)
            print "Iter ", iter
            # print "x = (%.2f, %.2f, %.2f), f(x) = %.2f" % (t[0], t[1], t[2], f(x, y, t,bias))
            print f(x, y, t)
            # print "Gradient: ", df(x, y, t,bias), "\n"
        iter += 1
    return t, f(x, y, t)


def make_x_y():
    '''
    Returns the entire training set
    build the X (NxM matrix) [Training Set]
    build the Y (NXM matrix) [One-Hot-Encoding]
    '''
    m = sum(len(M["train" + str(i)]) for i in range(10))  # training size.

    x = zeros((784, m))
    y = zeros((10, m))
    col_num = 0  # keep track of which column we are on in the X, Y matrix

    for i in range(10):  # iterate through the 10 numbers
        training_set = M["train" + str(i)]  # retrieve the full training set

        for j in range(len(training_set)):
            # place the image in the jth column
            x[:, col_num] = training_set[j]
            y[i, col_num] = 1  # "i" is our current number, set to 1 for
            # one hot encoding
            col_num += 1  # append the next column now

    return x, y

def make_x_y_subset(size):

    '''
    Returns a subset of the training set
    build the X (NxM matrix) [Training Set]
    build the Y (NXM matrix) [One-Hot-Encoding]
    @input size - size of a subset
    '''
    m = 10 * size  # training size. 10 images per number
    random.seed(0)
    randomlist = random.random(5000)  # 5000 random numbers

    x = zeros((784, m))
    y = zeros((10, m))
    col_num = 0  # keep track of which column we are on in the X, Y matrix

    for i in range(10):  # iterate through the 10 numbers
        j = 0
        usedlist = []  # keeps track of which images have been used, reset for
        # each training set number
        curr_rand = 0
        training_set = M["train" + str(i)]  # retrieve the full training set
        while (j < size):

            index = int(randomlist[curr_rand] * len(training_set))
            if index not in usedlist:
                x[:, col_num] = training_set[index]  # place the image in the jth column
                y[i, col_num] = 1  # "i" is our current number, set to 1 for
                # one hot encoding
                usedlist.append(index)
                j += 1
                col_num += 1  # append the next column now
            curr_rand += 1  # go to the next random index
    return x, y


def part4(alpha):
    '''
    Initialization of the gradient descent algorithm as specified
    :param alpha: step size
    :return: a tuple containing the optimized weights and the function value
    '''

    weights = zeros((785, 10))
    random.seed(3)
    xy = make_x_y()
    x = xy[0]
    y = xy[1]

    optimized_weights = grad_descent(f, df, x, y, weights, alpha)
    return optimized_weights


def part_4_test(optimized_weights):
    '''
    Tests performance on the training and test sets
    :param optimized_weights: thetas that will be tested
    :return: performance values in a tuple
    '''
    performance_train = 0
    performance_test = 0

    m = sum(len(M["train" + str(i)]) for i in range(10))  # training size.
    for i in range(10):
        results = part2(M["train" + str(i)].T, optimized_weights)
        for j in range(len(results.T)):  # tranpose the matrix so that it's 10xm
            # now we can loop through the rows
            y = argmax(results.T[j])
            if y == i:
                performance_train += 1

    n = sum(len(M["test" + str(i)]) for i in range(10))  # test size.
    for i in range(10):
        results = part2(M["test" + str(i)].T, optimized_weights)
        for j in range(len(results.T)):  # tranpose the matrix so that it's 10xm
            # now we can loop through the rows
            y = argmax(results.T[j])
            if y == i:
                performance_test += 1

    return performance_train / float(m), performance_test / float(n)


def plot_weights(learning_weights):
    '''
    Shows the learning curve and visualizes weights provided
    :param learning_weights: optimal weights found (array where the last entry contains the final iteration)
    :return: None
    '''
    # take the last weights, which corresponds to the final performance
    final_index = len(learning_weights) - 1
    final_weights = learning_weights[final_index][1:, :]  # 784 by 10
    # there are 10 weights, each of size 784, going to each of the outputs
    # we want to visualize all of them

    # unflatten the images, transpose them so we can index the row
    w1 = final_weights.T[0].reshape((28, 28))  # first row of 10x784
    w2 = final_weights.T[1].reshape((28, 28))
    w3 = final_weights.T[2].reshape((28, 28))
    w4 = final_weights.T[3].reshape((28, 28))
    w5 = final_weights.T[4].reshape((28, 28))
    w6 = final_weights.T[5].reshape((28, 28))
    w7 = final_weights.T[6].reshape((28, 28))
    w8 = final_weights.T[7].reshape((28, 28))
    w9 = final_weights.T[8].reshape((28, 28))
    w10 = final_weights.T[9].reshape((28, 28))

    # Four axes, returned as a 2-d array
    f, axarr = plt.subplots(2, 5)
    axarr[0, 0].imshow(w1, cmap=cm.gray)
    axarr[0, 1].imshow(w2, cmap=cm.gray)
    axarr[0, 2].imshow(w3, cmap=cm.gray)
    axarr[0, 3].imshow(w4, cmap=cm.gray)
    axarr[0, 4].imshow(w5, cmap=cm.gray)
    axarr[1, 0].imshow(w6, cmap=cm.gray)
    axarr[1, 1].imshow(w7, cmap=cm.gray)
    axarr[1, 2].imshow(w8, cmap=cm.gray)
    axarr[1, 3].imshow(w9, cmap=cm.gray)
    axarr[1, 4].imshow(w10, cmap=cm.gray)

    # Fine-tune figure; hide x ticks for top plots and y ticks for right plots
    plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
    plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)

    plt.show()
#--------------------------------------- Part 5 -----------------------------------------#
def part5():
    '''
    Demonstration of how logistic regression handles cases with more scattered data
    (i.e. the cost function won't be large when the outputs are off from the actual target ouputs...
    ... thus, the weights aren't adjusted too much)
    The function plots the actual data, least squares solution and the logistic regression (2D)
    :return: None
    '''
    # According to Piazza posts/answers, using the least square method and 2 labels is suitable for the demonstration
    N = 100
    sigma = 80 # Large sigma to generate datapoints far away from the target
    theta = array([0.3, 1])
    gen_lin_data_1d(theta, N, sigma)
    show()
    return

def gen_lin_data_1d(theta, N, sigma):
    '''
    Generates data according to the Gaussian distribution
    Plots the actual data, least squares solution and the logistic regression (2D)
    :param theta: actual line weights
    :param N: number of samples
    :param sigma: standard deviation
    :return: None
    '''
    #####################################################
    # Data limits
    x_limit = 150
    y_limit = 150

    # Actual data generation
    random.seed(N)
    x_raw = rint(x_limit * (random.random((N)) - .5)).astype(int64)
    x = vstack((ones_like(x_raw), x_raw,))
    y = dot(theta, x) + norm.rvs(scale=sigma, size=N)

    plot(x[1, :], y, "ro", label="Training set")

    # Actual generating process
    plot_line(theta, -x_limit, x_limit, "b", "Actual generating process")

    # Least squares solution
    theta_hat = dot(linalg.inv(dot(x, x.T)), dot(x, y.T))
    plot_line(theta_hat, -x_limit, x_limit, "g", "Maximum Likelihood Solution")

    # Logistic regression solution
    theta_log = grad_descent(f, df, np.reshape(x_raw, (1,N)), np.reshape(y, (1,N)), ones((2, 1)), 0.00000001)[0].T
    theta_log = np.reshape(theta_log, (2,))

    plot_line(theta_log, -x_limit, y_limit, "r", "Logistic Regression Solution")

    legend(loc=4)
    xlim([-x_limit, x_limit])
    ylim([-y_limit, y_limit])
    plt.savefig("logistic_vs_least_squares.png")

def plot_line(theta, x_min, x_max, color, label):
    '''
    The function plots the line
    :param theta: line coefficients
    :param x_min: minimum x
    :param x_max: maximum x
    :param color: color of the line
    :param label: label of the line
    :return: None
    '''
    x_grid_raw = arange(x_min, x_max, 0.01)
    x_grid = vstack((ones_like(x_grid_raw), x_grid_raw,))
    y_grid = dot(theta, x_grid)
    plot(x_grid[1,:], y_grid, color, label=label)

#---------------------------------- Helper functions given on the 411 website ------------------------------------#

def tanh_layer(y, W, b):
    '''Return the output of a tanh layer for the input matrix y. y
    is an NxM matrix where N is the number of inputs for a single case, and M
    is the number of cases'''
    return tanh(dot(W.T, y) + b)


def forward(x, W0, b0, W1, b1):
    L0 = tanh_layer(x, W0, b0)
    L1 = dot(W1.T, L0) + b1
    output = softmax(L1)
    return L0, L1, output


def NLL(y, y_):
    return -sum(y_ * log(y))


def deriv_multilayer(W0, b0, W1, b1, x, L0, L1, y, y_):
    '''Incomplete function for computing the gradient of the cross-entropy
    cost function w.r.t the parameters of a neural network'''
    dCdL1 = y - y_
    dCdW1 = dot(L0, dCdL1.T)


'''
#Load sample weights for the multilayer neural network
snapshot = cPickle.load(open("snapshot50.pkl"))
W0 = snapshot["W0"]
b0 = snapshot["b0"].reshape((300,1))
W1 = snapshot["W1"]
b1 = snapshot["b1"].reshape((10,1))

#Load one example from the training set, and run it through the
#neural network
x = M["train5"][148:149].T
L0, L1, output = forward(x, W0, b0, W1, b1)
#get the index at which the output is the largest
y = argmax(output)
'''
################################################################################
# Code for displaying a feature from the weight matrix mW
# fig = figure(1)
# ax = fig.gca()
# heatmap = ax.imshow(mW[:,50].reshape((28,28)), cmap = cm.coolwarm)
# fig.colorbar(heatmap, shrink = 0.5, aspect=5)
# show()
################################################################################


if run_part1:
    for i in range(10):
        plot_figures("train" + str(i))

if run_part3:
    part3()

if run_part4:
    #alpha = 0.000001
    # optimized_weights = part4(alpha)
    # optimized_weights = optimized_weights[0]
    # part_4_test(optimized_weights)
    '''
    Uncomment above if we just want to see the accuracy on the entire training set and test set, for larger alpha = 0.000001
    With this alpha:

    Performance on the training set: 0.917016666667
    Performance on the test set: 0.9194
    '''
    alpha = 0.0000001
    # A .pkl file with learning_weights is submitted
    if os.path.isfile('learning_weights' + str(alpha) +'.pkl'):
        print("Loading from the .pkl file...")
        learning_weights = reload_weights(alpha)
    else:
        part4(alpha)
        save_weights(learning_weights, alpha)

    # Plot the learning curve obtained through training:
    results = [(part_4_test(learning_weights[i])[0], part_4_test(learning_weights[i])[1]) for i in
               range(len(learning_weights))]
    x_axis = linspace(0, 3000, len(results))
    training_results = [results[i][0] for i in range(len(results))]
    test_results = [results[i][1] for i in range(len(results))]

    plt_training = plt.plot(x_axis, training_results, label='Training')
    plt_test = plt.plot(x_axis, test_results, label='Test')
    plt.ylim([0.6, 0.9])
    plt.xlabel('# of Iterations')
    plt.ylabel('Performance')
    plt.title('Performance vs. # of iterations')
    plt.legend(["Training Set", "Test Set"], loc=7)
    plt.savefig("learning_curve.png")

    # Report on the final performance:
    final_index = len(learning_weights) - 1
    final_result = part_4_test(learning_weights[final_index])
    print("Performance on the training set: " + str(final_result[0]))
    print("Performance on the test set: " + str(final_result[1]))
    plot_weights(learning_weights)

if run_part5:
    part5()
    print("--in progress--")


