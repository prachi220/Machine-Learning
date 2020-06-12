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
import math
from numpy import exp, array, random, dot
from numpy import genfromtxt
from sklearn import linear_model, datasets
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np

def normalise(x):
	n = 2
	m = len(x)
	for j in range(0,n):
		u=0
		for i in xrange(0,m):
			u += x[i][j]
		u = u/m
		var = 0
		for i in xrange(0,m):
			var += (x[i][j]-u)*(x[i][j]-u)
		var =var/m
		sigma = math.sqrt(var)
		for i in xrange(0,m):
			x[i][j] = ((x[i][j]-u)/sigma)
	return x


X1 = np.loadtxt("toy_data/toy_trainX.csv", dtype=float, delimiter=',')
Y = np.loadtxt("toy_data/toy_trainY.csv", dtype=float, delimiter=',')

Xt1 = np.loadtxt("toy_data/toy_testX.csv", dtype=float, delimiter=',')
Yt = np.loadtxt("toy_data/toy_testY.csv", dtype=float, delimiter=',')
h = .02  # step size in the mesh
logreg = linear_model.LogisticRegression(C=1e5)
# we create an instance of Neighbours Classifier and fit the data.
logreg.fit(X1, Y)
#train prediction score
logreg.score(X1,Y)
#0.45789473684210524
# test prediction score
logreg.score(Xt1,Yt)
# 0.38333333333333336

def plot_decision_boundary(model, X, y):
    """
    Given a model(a function) and a set of points(X), corresponding labels(y), scatter the points in X with color coding
    according to y. Also use the model to predict the label at grid points to get the region for each label, and thus the 
    descion boundary.
    Example usage:
    say we have a function predict(x,other params) which makes 0/1 prediction for point x and we want to plot
    train set then call as:
    plot_decision_boundary(lambda x:predict(x,other params),X_train,Y_train)
    params(3): 
        model : a function which expectes the point to make 0/1 label prediction
        X : a (mx2) numpy array with the points
        y : a (mx1) numpy array with labels
    outputs(None)
    """
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = 0.02
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.show()

# def model(x_t):
# 	predictions = logisticRegr.predict(x_t)
# 	print predictions.shape
# 	return predictions

# print x_train.shape
# print y_train.shape
# plot_decision_boundary(logisticRegr.predict,x_train,y_train)
# plot_decision_boundary(logreg.predict,X1,Y)
plot_decision_boundary(logreg.predict,Xt1,Yt)
