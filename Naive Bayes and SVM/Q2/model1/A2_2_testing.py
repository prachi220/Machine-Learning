import sys
import math
import string
from sets import Set
from random import randint
import time
import pickle
import numpy as np
import pandas as pd
import csv
from collections import Counter

class classifier:
    def __init__(self, w,b,cl1,cl2):
        self.w = w
        self.b = b
        self.cls1 = cl1
        self.cls2 = cl2

f = open("Q2/model1/svm", "r")
svm = pickle.load(f)
f.close()
classifiers = list(svm)[0]
cnt  = list(svm)[1]

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

def find_mode(l):
	max_cnt = 0
	max_class = -1
	num_class = dict()
	for x in xrange(0,len(l)):
		cls = l[x]
		if num_class.get(cls) is None:
			num_class[cls] = 0
		num_class[cls] += 1

	for i in num_class:
		cnt = num_class[i]
		if cnt > max_cnt:
			max_cnt = cnt
			max_class = i
		if cnt == max_cnt:
			if i > max_class:
				max_class = i

	return max_class

def writefile(out_file,Y):
	with open(out_file, 'w') as out:
		for i in xrange(len(Y)):
			out.write(str(Y[i]))
			if (i<len(Y)-1):
				out.write('\n')

ConfusionMatrix = {}
for i in range(0,10):
	ConfusionMatrix[i] = {}
	for j in range(0,10):
		(ConfusionMatrix[i])[j] = 0

Xt, Yt, classes_t = read_data(sys.argv[1])
outY = {}
correct_count = 0.0
total_count = 0.0
for i in xrange(0,len(Xt)):
	total_count += 1
	x = Xt[i]
	y = Yt[i]
	out_classes = {}
	for k in xrange(0,cnt):
		clf = classifiers[k]
		w = clf.w
		b = clf.b
		cl1 = clf.cls1
		cl2 = clf.cls2
		if (np.dot(w,x.T)+b >= 0):
			out_classes[k] = cl1
		if (np.dot(w,x.T)+b < 0):
			out_classes[k] = cl2
	class_name = find_mode(out_classes)
	outY[i] = class_name
	if class_name == y:
		correct_count += 1

writefile(sys.argv[2],outY)

print correct_count
print total_count
print (correct_count/total_count)*100
