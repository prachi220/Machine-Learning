import numpy as np
np.seterr(divide='ignore',invalid='ignore')
import itertools
import random
import sys
import time
from sklearn import svm
from collections import Counter
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans  
import cPickle as pickle
np.set_printoptions(threshold=np.inf)

train_labels = {}
count = 0
ex0 = np.load('train/harp.npy')
train_data = ex0
for j in range(0,5000):
	train_labels[count] = 0
	count += 1

ex1 = np.load('train/banana.npy')
train_data = np.append(train_data,ex1,axis=0)
for j in range(0,5000):
	train_labels[count] = 1
	count += 1

ex2 = np.load('train/bulldozer.npy')
train_data = np.append(train_data,ex2,axis=0)
for j in range(0,5000):
	train_labels[count] = 2
	count += 1

ex3 = np.load('train/chair.npy')
train_data = np.append(train_data,ex3,axis=0)
for j in range(0,5000):
	train_labels[count] = 3
	count += 1

ex4 = np.load('train/eyeglasses.npy')
train_data = np.append(train_data,ex4,axis=0)
for j in range(0,5000):
	train_labels[count] = 4
	count += 1

ex5 = np.load('train/flashlight.npy')
train_data = np.append(train_data,ex5,axis=0)
for j in range(0,5000):
	train_labels[count] = 5
	count += 1

ex6 = np.load('train/foot.npy')
train_data = np.append(train_data,ex6,axis=0)
for j in range(0,5000):
	train_labels[count] = 6
	count += 1

ex7 = np.load('train/hand.npy')
train_data = np.append(train_data,ex7,axis=0)
for j in range(0,5000):
	train_labels[count] = 7
	count += 1

ex8 = np.load('train/hat.npy')
train_data = np.append(train_data,ex8,axis=0)
for j in range(0,5000):
	train_labels[count] = 8
	count += 1

ex9 = np.load('train/keyboard.npy')
train_data = np.append(train_data,ex9,axis=0)
for j in range(0,5000):
	train_labels[count] = 9
	count += 1

ex10 = np.load('train/laptop.npy')
train_data = np.append(train_data,ex10,axis=0)
for j in range(0,5000):
	train_labels[count] = 10
	count += 1

ex11 = np.load('train/nose.npy')
train_data = np.append(train_data,ex11,axis=0)
for j in range(0,5000):
	train_labels[count] = 11
	count += 1

ex12 = np.load('train/parrot.npy')
train_data = np.append(train_data,ex12,axis=0)
for j in range(0,5000):
	train_labels[count] = 12
	count += 1

ex13 = np.load('train/penguin.npy')
train_data = np.append(train_data,ex13,axis=0)
for j in range(0,5000):
	train_labels[count] = 13
	count += 1

ex14 = np.load('train/pig.npy')
train_data = np.append(train_data,ex14,axis=0)
for j in range(0,5000):
	train_labels[count] = 14
	count += 1

ex15 = np.load('train/skyscraper.npy')
train_data = np.append(train_data,ex15,axis=0)
for j in range(0,5000):
	train_labels[count] = 15
	count += 1

ex16 = np.load('train/snowman.npy')
train_data = np.append(train_data,ex16,axis=0)
for j in range(0,5000):
	train_labels[count] = 16
	count += 1

ex17 = np.load('train/spider.npy')
train_data = np.append(train_data,ex17,axis=0)
for j in range(0,5000):
	train_labels[count] = 17
	count += 1

ex18 = np.load('train/trombone.npy')
train_data = np.append(train_data,ex18,axis=0)
for j in range(0,5000):
	train_labels[count] = 18
	count += 1

ex19 = np.load('train/violin.npy')
train_data = np.append(train_data,ex19,axis=0)
for j in range(0,5000):
	train_labels[count] = 19
	count += 1

f = open("data", "w")
pickle.dump([train_data, train_labels], f)
f.close()