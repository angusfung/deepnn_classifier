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


## Running The Code.
run_part1 = True



#Load the MNIST digit data
M = loadmat("mnist_all.mat")

# #Display the 150-th "5" digit from the training set
# imshow(M["train1"][120].reshape((28,28)), cmap=cm.gray)
# show()



    
def plot_figures(filename):
    '''@param filename e.g "train1", "train2"
    '''
    #generates 10 rand numbers 
    random.seed(1)
    rand = random.random(10)*len(M[filename]) 

    #unflatten the images    
    x1 = M[filename][int(rand[0])].reshape((28,28))
    x2 = M[filename][int(rand[1])].reshape((28,28))
    x3 = M[filename][int(rand[2])].reshape((28,28)) 
    x4 = M[filename][int(rand[3])].reshape((28,28))
    x5 = M[filename][int(rand[4])].reshape((28,28)) 
    x6 = M[filename][int(rand[5])].reshape((28,28)) 
    x7 = M[filename][int(rand[6])].reshape((28,28)) 
    x8 = M[filename][int(rand[7])].reshape((28,28)) 
    x9 = M[filename][int(rand[8])].reshape((28,28)) 
    x10 = M[filename][int(rand[9])].reshape((28,28)) 
    
    # Four axes, returned as a 2-d array
    f, axarr = plt.subplots(2, 5)
    axarr[0, 0].imshow(x1,cmap=cm.gray)
    axarr[0, 1].imshow(x2,cmap=cm.gray)
    axarr[0, 2].imshow(x3,cmap=cm.gray)
    axarr[0, 3].imshow(x4,cmap=cm.gray)
    axarr[0, 4].imshow(x5,cmap=cm.gray)
    axarr[1, 0].imshow(x6,cmap=cm.gray)
    axarr[1, 1].imshow(x7,cmap=cm.gray)
    axarr[1, 2].imshow(x8,cmap=cm.gray)
    axarr[1, 3].imshow(x9,cmap=cm.gray)
    axarr[1, 4].imshow(x10,cmap=cm.gray)
    
    # Fine-tune figure; hide x ticks for top plots and y ticks for right plots
    plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
    plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)
    
    plt.show()


def softmax(y):
    '''Return the output of the softmax function for the matrix of output y. y
    is an NxM matrix where N is the number of outputs for a single case, and M
    is the number of cases'''
    
    return exp(y)/tile(sum(exp(y),0), (len(y),1))
    
def part2(x,weights,bias):
    '''returns a matrix of probabiltiies, the output from the cost function'''
    #y = sum(dot(weights.T, x),1) + bias
    
    x = x/255.0
    
    x = vstack( (ones((1, x.shape[1])), x)) #include the bias terms
    weights = vstack( (ones((1, weights.shape[1])), weights))
    
    y = dot(weights.T, x) + bias
    return softmax(y)
    
def f(x,y,weights,bias):
    '''@input prob is a NxM matrix of probabilities
       @input y is a NxM matrix where each column is the one-hot-representation         
              of an image  
              N is the number of outputs for a single case, M is the number of 
              cases (e.g images)     
              '''
    prob=part2(x,weights,bias)
    return -sum(y * log(prob))
 '''
#alternative cost function, but high run-time complexity (more intuitive)
def f1(x,y,weights,bias):
    sum=0
    prob=part2(x,weights,bias)
    for j in range(10):
        for k in range(200):
            sum += y[j][k] * log(prob[j][k])
    return -sum
'''
    
def df(x, y, weights, bias):
    prob=part2(x,weights,bias)
    x=x/255.0
    return dot(x,(prob-y).T)

 
def make_x_y_subset(size): #takes a subset of the training set 
                           #probably don't need this code, because guerzhoy
                           #made a clarification.
    '''build the X (NxM matrix) [Training Set]
       build the Y (NXM matrix) [One-Hot-Encoding]
       @input size is the number of images of each digit in the trainingset.
    '''
    m=10*size #training size. 10 images per number
    random.seed(0)
    randomlist = random.random(5000)  #5000 random numbers
    
    x = zeros((784,m))
    y = zeros((784,m))
    col_num = 0 #keep track of which column we are on in the X, Y matrix
    
    for i in range(10): #iterate through the 10 numbers
        j=0
        usedlist = [] #keeps track of which images have been used, reset for 
                      #each training set number 
        curr_rand = 0
        training_set = M["train" + str(i)] #retrieve the full training set
        while (j<size):
            
            index = int(randomlist[curr_rand] *len(training_set)) 
            if index not in usedlist:
                x[:,col_num] = training_set[index] #place the image in the jth column           
                y[i,col_num] = 1 # "i" is our current number, set to 1 for
                                 # one hot encoding
                usedlist.append(index)
                j+=1
                col_num += 1 #append the next column now
            curr_rand += 1 #go to the next random index
    return x,y

def make_x_y(): #takes the entire trainingset (randomoness not required) 
    '''build the X (NxM matrix) [Training Set]
       build the Y (NXM matrix) [One-Hot-Encoding]
       @input size is the number of images of each digit in the trainingset.
    '''
    m = sum(len(M["train" + str(i)]) for i in range(10)) #training size.
    
    x = zeros((784,m))
    y = zeros((784,m))
    col_num = 0 #keep track of which column we are on in the X, Y matrix
    
    for i in range(10): #iterate through the 10 numbers
        training_set = M["train" + str(i)] #retrieve the full training set

        for j in range(len(training_set)):
            #place the image in the jth column 
            x[:,col_num] = training_set[j] 
            y[i,col_num] = 1 # "i" is our current number, set to 1 for
                                 # one hot encoding
            col_num += 1 #append the next column now

    return x,y
    
##PARAMETERS FOR GRADIENT DESCENT
#initializing random variables to test  

m=200

random.seed(0)
x = reshape(random.rand(784*m), (784,m))
random.seed(1)
y=zeros((10,m)) #one-hot encoding matrix
y[0,:]=1
random.seed(2)
weights = reshape(random.rand(784*10), (784,10))
random.seed(3)
bias = reshape(random.rand(10),(10,1))

h = 0.0000001

random.seed(4)
dh = zeros((784,10))
dh[0,0]=h

print (f(x,y, weights+dh,bias) - f(x,y, weights-dh,bias))/(2*h)
print df(x, y, weights,bias)




def grad_descent(f, df, x, y, init_t, alpha,bias):
    EPS = 1e-10   #EPS = 10**(-5)
    prev_t = init_t-10*EPS
    t = init_t.copy()
    max_iter = 60000
    iter  = 0
    while norm(t - prev_t) >  EPS and iter < max_iter:
        prev_t = t.copy()
        t -= alpha*df(x, y, t,bias)
        if iter % 2000 == 0:
        #if iter % 1 == 0:
            print "Iter", iter
            print "x = (%.2f, %.2f, %.2f), f(x) = %.2f" % (t[0], t[1], t[2], f(x, y, t,bias)) 
            print "Gradient: ", df(x, y, t,bias), "\n"
        iter += 1
    return t,f(x,y,t,bias)

#optimized_weights = grad_descent(f,df,x,y,weights, 0.0000000001, bias)








def tanh_layer(y, W, b):    
    '''Return the output of a tanh layer for the input matrix y. y
    is an NxM matrix where N is the number of inputs for a single case, and M
    is the number of cases'''
    return tanh(dot(W.T, y)+b)

def forward(x, W0, b0, W1, b1):
    L0 = tanh_layer(x, W0, b0)
    L1 = dot(W1.T, L0) + b1
    output = softmax(L1)
    return L0, L1, output
    
def NLL(y, y_):
    return -sum(y_*log(y)) 

def deriv_multilayer(W0, b0, W1, b1, x, L0, L1, y, y_):
    '''Incomplete function for computing the gradient of the cross-entropy
    cost function w.r.t the parameters of a neural network'''
    dCdL1 =  y - y_
    dCdW1 =  dot(L0, dCdL1.T ) 
    
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
#Code for displaying a feature from the weight matrix mW
#fig = figure(1)
#ax = fig.gca()    
#heatmap = ax.imshow(mW[:,50].reshape((28,28)), cmap = cm.coolwarm)    
#fig.colorbar(heatmap, shrink = 0.5, aspect=5)
#show()
################################################################################


if run_part1:
    for i in range(10):
        plot_figures("train" + str(i))