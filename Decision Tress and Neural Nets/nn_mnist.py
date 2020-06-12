import sys
import math
import string
from sets import Set
from random import randint
import cPickle as pickle
import numpy as np
import pandas as pd
import csv
import random
from numpy import exp, array, random, dot
from numpy import genfromtxt

def sigmoid(x):
    e1 = 1 + exp(-x)
    e2 = 1/e1
    # print e2.shape
    return e2

def random_weight(lout, lin):
    eps_init = (np.sqrt(6))/(np.sqrt(lout) + np.sqrt(lin))
    ans = np.matrix(np.zeros((lout, lin)))
    np.random.seed(1)
    ans = (np.random.randn(lout, lin) * 2 * eps_init) - eps_init
    # ans = random.random((lout, lin))*0.01
    return ans

def back_propagation(layers,layers_output,trainY,trainX):
    n = len(layers)
    layers_delta = {}
    weight_change = {}
    x0 = layers_output[n-1]
    layers_delta[n-1] = np.asarray((-1)* np.array(trainY-x0)*x0*(1-x0))
    for i in xrange(2,n+1):
        x1 = np.asarray(layers_output[n-i])
        y1 = layers_delta[n-i+1].dot(layers[n-i+1].T)
        layers_delta[n-i] = np.asarray(y1*x1*(1-x1))
    y2 = np.zeros((1,len(layers_delta[0])),dtype=np.float64)
    y2[0,:] = layers_delta[0]
    x2 = np.zeros((len(trainX.T),1),dtype=np.float64)
    x2[:,0] = trainX.T
    weight_change[0] = x2.dot(y2) 
    for i in xrange(1,n):
        y = np.zeros((1,len(layers_delta[i])),dtype=np.float64)
        y[0,:] = layers_delta[i]
        x = np.zeros((len(layers_output[i-1]),1),dtype=np.float64)
        x[:,0] = layers_output[i-1]
        weight_change[i] = x.dot(y)   
    return weight_change

def error(h_theta, trainY):
    error = 0.0
    for i in range(0,len(trainY)):
        # print "trainY: ", trainY[i][0]
        # print "h_theta: ", h_theta[i][0]
        x = (trainY[i][0]-h_theta[i])
        error += x*x
    error = error/2
    return error

def test(test_data, layers):
    layers_output = {}
    layers_output[0] = sigmoid(dot(test_data, layers[0]))
    for k in range(1,len(layers)):
        layers_output[k] = sigmoid(dot(layers_output[k-1], layers[k]))
    return layers_output

def get_accuracy(output, label):
    accuracy = 0.0
    if output[len(layers)-1] > 0.5:
        if label==1:
            accuracy = 1
    else:
        if label==0:
            accuracy = 1
    return accuracy

def train_sgd(trainX, trainY, layers, batch_size):
    np.random.seed(1)
    m, n = trainX.shape
    print m,n
    data = np.zeros((m,n+1),dtype=np.float64)
    for i in range(0,m):
        for j in range(0,n):
            data[i][j] = trainX[i][j]
        data[i][n] = trainY[i]
    np.random.shuffle(data)
    X_train = np.array(data[:,0:n])
    print "X",X_train.shape
    Y_train = np.array([data[:,n]]).T
    print "Y",Y_train.shape
    num_iter = 1000
    layers_output = {}
    print "Starting to train..."
    # for j in range(num_iter):
    dec = 0
    itern = 0
    err = 1000
    # alpha = 0.03
    X = X_train[batch_size:m,:]
    Y = Y_train[batch_size:m,:]
    prev_accuracy = 0.0
    for i in range(0,len(X)):
        out = test(X[i],layers)
        # h_theta.append(out[len(layers)-1])
        prev_accuracy += get_accuracy(out,Y[i])
    while 1:
        itern += 1
        alpha = 0.1/math.sqrt(itern)
        for i in range(0, batch_size):
            layers_output[0] = sigmoid(np.dot(X_train[i], layers[0]))
            for k in range(1,len(layers)):
                layers_output[k] = sigmoid(np.dot(layers_output[k-1], layers[k]))
            # print "layers_output", len(layers_output), len(layers_output[0])
            weight_change = back_propagation(layers,layers_output,Y_train[i], X_train[i])
            # print "########################## ITERATION ", i, " ############################"
            for k in range(0,len(layers)):
                layers[k] -= alpha*weight_change[k]
                # print layers[k]
        accuracy = 0.0
        h_theta = []
        for i in range(0,len(X)):
            out = test(X[i],layers)
            h_theta.append(out[len(layers)-1])
            accuracy += get_accuracy(out,Y[i])
        err = error(h_theta,Y)
        print "error : ",err
        print "accuracy : ", accuracy
        # if itern == 1000:
        #     break;
        if accuracy > prev_accuracy:
            if dec == 20:
                break;
            else:
                dec += 1
            
        else:
            prev_accuracy = accuracy

def read_data(filename):
    dataX = pd.read_csv(filename, header=None)
    X1 = np.array(dataX.as_matrix(columns=None))
    m, n1 = X1.shape
    n = n1-1
    Y = {}
    classes = set()
    cnt = 0
    X = np.zeros((m,n),dtype=np.float64)
    for i in range(0,m):
        p = X1[i][n1-1]
        Y[i] = p
        if p not in classes:
            classes.add(p)      
        for j in range(0,n):
            X[i][j] = X1[i][j]
        min_i = min(X[i])
        for j in range(0,n):
            X[i][j] = X[i][j]-min_i
        max_i = max(X[i])   
        for j in range(0,n):
            X[i][j] = X[i][j]/max_i
    return X, Y, classes

if __name__ == "__main__":
    train_data, train_label, classes = read_data("mnist_data/MNIST_train.csv") 
    for i in range(0,len(train_label)):
        if train_label[i]==6:
            train_label[i]=0
        else:
            train_label[i]=1
    test_data, test_label, classes1 = read_data("mnist_data/MNIST_test.csv")
    for i in range(0,len(test_label)):
        if test_label[i]==6:
            test_label[i]=0
        else:
            test_label[i]=1
    num_inputs = len(train_data)
    num_features = len(train_data[0])
    layer_array = [1]
    batch_size = 100
    num_layers = len(layer_array)
    layers = {}
    for i in xrange(0,num_layers):
        # print layer_array[i]
        l = (int)(layer_array[i])
        layers[i] = random_weight(num_features,l)
        num_features = l
    print layers
    for i in xrange(0,len(layers)):
        print layers[i]
    train_sgd(train_data, train_label, layers, batch_size)
    for i in xrange(0,len(layers)):
        print layers[i]
    f = open("neural_nets_mnist", "w")
    pickle.dump([layers], f)
    f.close()
    