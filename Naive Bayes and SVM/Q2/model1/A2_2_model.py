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
from collections import Counter

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

def create_class_pairs(X,Y,cl1,cl2):
	m,n = X.shape
	X_1 = {}
	Y_1 = []
	x_cnt = 0
	y_cnt = 0
	for k in xrange(0,m):
		if Y[k] == cl1:
			X_1[x_cnt] = X[k]
			Y_1.append(1)
			x_cnt += 1
			y_cnt += 1
		if Y[k] == cl2:
			X_1[x_cnt] = X[k]
			Y_1.append(-1)
			x_cnt += 1
			y_cnt += 1
	m1 = len(Y_1)
	# print len(X_1)
	X1_arr = np.zeros((m1,n),dtype=np.float64)
	for x in range(0,m1):
		for y in range(0,n):
			X1_arr[x][y] = X_1[x][y]
	Y1_arr = np.array(Y_1)
	return X1_arr, Y1_arr, n, m1

def pegasos(X,Y,l,n,m,itr,k):
	w = np.zeros((1,n),dtype=np.float64)
	C = 1
	# print m, n
	b = 0.0
	t = 0
	diff1 = sys.maxint
	diff2 = sys.maxint
	# for t in range(1,itr+1):
	while diff1 > 0.001:
		t += 1
		A = random.sample(range(0,m),k)
		total = np.zeros((1,n),dtype=np.float64)
		total2 = 0.0
		cnt = 0
		eta = 1.0/(l*t)
		for i in A:
			x = X[i]
			y = Y[i]
			p = y*((np.dot(w,x.T))+b)
			if p < 1:
				p1 = y*x
				total = np.add(total,p1)
				total2 = total2+y
		w1 = np.add((w*(1.0-(1.0/t))) , (C*eta*total))
		b = b + ((eta/k)*total2)
		diff1 = np.linalg.norm(np.subtract(w,w1))
		w = w1
	return w, b,t

class classifier:
    def __init__(self, w,b,cl1,cl2):
        self.w = w
        self.b = b
        self.cls1 = cl1
        self.cls2 = cl2

X, Y, classes = read_data('train.csv')
classes = list(classes)
classifiers = dict()
cnt = 0

for i in xrange(0,len(classes)):
	cl1 = classes[i]
	for j in xrange(i+1,len(classes)):
		cl2 = classes[j]
		X_arr, Y_arr, n, m1 = create_class_pairs(X,Y,cl1,cl2)
		w,b,t = pegasos(X_arr,Y_arr,1,n,m1,10000,100)
		classifiers[cnt] = classifier(w,b,cl1,cl2)
		cnt += 1


f = open("svm", "w")
pickle.dump([classifiers,cnt], f)
f.close()

