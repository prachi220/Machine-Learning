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

def read_data(filename):
	dataX = pd.read_csv(filename, header=None)
	X1 = np.array(dataX.as_matrix(columns=None))
	m, n1 = X1.shape
	n = n1-1
	Y = {}
	classes = set()
	count_class = dict()
	cnt = 0
	X = np.zeros((m,n),dtype=np.float64)
	for i in range(0,m):
		p = X1[i][n1-1]
		Y[i] = p
		if p not in classes:
			classes.add(p)
			count_class[p] = 0.0
		count_class[p] += 1		
		for j in range(0,n):
			X[i][j] = X1[i][j]
		min_i = min(X[i])
		for j in range(0,n):
			X[i][j] = X[i][j]-min_i
		max_i = max(X[i])	
		for j in range(0,n):
			X[i][j] = X[i][j]/max_i
	print count_class
	return X, Y, classes

def libSVMformat(X,Y, out_file):
    with open(out_file, 'w') as out:
        for i in xrange(len(X)):
            out.write(str(Y[i]))
            for j in xrange(1,len(X[i])+1):
                if X[i][j-1] != 0:
                    out.write(str(' ')) 
                    out.write(str(j)+':'+str(X[i][j-1]))
            if (i<len(X)-1):
                out.write('\n')

X, Y, classes = read_data(sys.argv[1])
libSVMformat(X,Y,'libsvm_test.txt')
