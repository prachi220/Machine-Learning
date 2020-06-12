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
import csv 
import cPickle as pickle
np.set_printoptions(threshold=np.inf)

f1 = open("k_means", "r")
forward = pickle.load(f1)
f1.close()
kmeans = list(forward)[0]
train_data = list(forward)[1]
train_labels = list(forward)[2]
test_data = np.load('test/test.npy')
test_data = (test_data - test_data.mean(axis=0)) / (test_data.std(axis=0))
def getLabelsMap(labels):
	labels_map = {}
	for j in range(20):
	    labs = {}
	    for jj in range(20):
	        labs[jj] = 0
	    indices = [i for i in range(100000) if labels[i] == j]
	    for h in indices:
	        g = h/5000 // 1
	        labs[g] = labs[g] + 1
	    max_amt = 0
	    max_lab = 0
	    for u in range(20):
	        if labs[u] > max_amt:
	            max_amt = labs[u]
	            max_lab = u
	    labels_map[j] = max_lab
	return labels_map

predict_labels_train = kmeans.labels_
predict_labels_test = kmeans.predict(test_data)

labels_map = getLabelsMap(predict_labels_train)
for i in range(0,len(predict_labels_train)):
	lab1 = predict_labels_train[i]
	lab2 = predict_labels_test[i]
	predict_labels_train[i] = labels_map[lab1]
	predict_labels_test[i] = labels_map[lab2]

train_accuracy = 0.0	
for i in range(0,len(train_labels)):
	if (predict_labels_train[i]==train_labels[i]):
		train_accuracy += 1
print train_accuracy

labels_class = {}
labels_class[0] = 'harp'
labels_class[1] = 'banana'
labels_class[2] = 'bulldozer'
labels_class[3] = 'chair'
labels_class[4] = 'eyeglasses'
labels_class[5] = 'flashlight'
labels_class[6] = 'foot'
labels_class[7] = 'hand'
labels_class[8] = 'hat'
labels_class[9] = 'keyboard'
labels_class[10] = 'laptop'
labels_class[11] = 'nose'
labels_class[12] = 'parrot'
labels_class[13] = 'penguin'
labels_class[14] = 'pig'
labels_class[15] = 'skyscraper'
labels_class[16] = 'snowman'
labels_class[17] = 'spider'
labels_class[18] = 'trombone'
labels_class[19] = 'violin'

print test_data.shape
print predict_labels_test.shape

with open('test_a.csv', 'w') as outfile:
	writer = csv.writer(outfile)
	writer.writerow(['ID', 'CATEGORY'])
	for i in range(0,len(predict_labels_test)):
		lab = predict_labels_test[i]
		writer.writerow([str(i),labels_class[lab]])