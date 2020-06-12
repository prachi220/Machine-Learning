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
    print e2.shape
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
    # print "layers_output", layers_output
    x0 = layers_output[n-1]
    layers_delta[n-1] = np.asarray((-1)* np.array(trainY-x0)*x0*(1-x0))
    # print "layers_delta[n-1]", layers_delta[n-1]
    for i in xrange(2,n+1):
        x1 = np.asarray(layers_output[n-i])
        y1 = layers_delta[n-i+1].dot(layers[n-i+1].T)
        layers_delta[n-i] = np.asarray(y1*x1*(1-x1))
    # print "layers_delta[0]", (np.asarray(layers_delta[0])).shape
    # print "layers_delta[0]", layers_delta[0]
    y2 = np.zeros((1,len(layers_delta[0])),dtype=np.float64)
    y2[0,:] = layers_delta[0]
    x2 = np.zeros((len(trainX.T),1),dtype=np.float64)
    x2[:,0] = trainX.T
    # print y.shape
    weight_change[0] = x2.dot(y2) 
    for i in xrange(1,n):
        y = np.zeros((1,len(layers_delta[i])),dtype=np.float64)
        y[0,:] = layers_delta[i]
        x = np.zeros((len(layers_output[i-1]),1),dtype=np.float64)
        x[:,0] = layers_output[i-1]
        weight_change[i] = x.dot(y)   
    return weight_change


def train_sgd(trainX, trainY, layers, batch_size):
    np.random.seed(1)
    alpha = 0.03
    m, n = trainX.shape
    print m,n
    data = np.zeros((m,n+1),dtype=np.float64)
    for i in range(0,m):
        for j in range(0,n):
            data[i][j] = trainX[i][j]
        data[i][n] = trainY[i]
    # data[:,0:n] = trainX
    # data[:,n+1] = trainY
    np.random.shuffle(data)
    X_train = np.array(data[:,0:n])
    print "X",X_train.shape
    Y_train = np.array([data[:,n]]).T
    print "Y",Y_train.shape
    num_iter = m*(40/27) 
    layers_output = {}
    print "Starting to train..."
    for j in range(num_iter):
        for i in range(0, batch_size):

            layers_output[0] = sigmoid(np.dot(X_train[i], layers[0]))
            for k in range(1,len(layers)):
                layers_output[k] = sigmoid(np.dot(layers_output[k-1], layers[k]))
            # print "layers_output", len(layers_output), len(layers_output[0])
            weight_change = back_propagation(layers,layers_output,Y_train[i], X_train[i])
            print "########################## ITERATION ", i, " ############################"
            for k in range(0,len(layers)):
                layers[k] -= alpha*weight_change[k]
                print layers[k]


if __name__ == "__main__":
    trainX = pd.read_csv("toy_data/toy_trainX.csv", header=None)
    train_data = np.array(trainX.as_matrix(columns=None))
    trainY = pd.read_csv("toy_data/toy_trainY.csv", header=None)
    train_label = np.array(trainY.as_matrix(columns=None))

    testX = pd.read_csv("toy_data/toy_testX.csv", header=None)
    test_data = np.array(testX.as_matrix(columns=None))
    testY = pd.read_csv("toy_data/toy_testY.csv", header=None)
    test_label = np.array(testY.as_matrix(columns=None))
    print train_data
    print train_label
    #Seed the random number generator
    num_inputs = len(train_data)
    num_features = len(train_data[0])
    layer_array = [10,1]
    batch_size = (int)(len(train_data))
    num_layers = len(layer_array)
    layers = {}
    for i in xrange(0,num_layers):
        print layer_array[i]
        l = (int)(layer_array[i])
        layers[i] = random_weight(num_features,l)
        num_features = l

    for i in xrange(0,len(layers)):
        print layers[i]
    train_sgd(train_data, train_label, layers, batch_size)
    for i in xrange(0,len(layers)):
        print layers[i]
    f = open("neural_nets", "w")
    pickle.dump([layers,train_data,train_label,test_data,test_label], f)
    f.close()
    