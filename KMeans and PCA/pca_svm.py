import numpy as np
np.seterr(divide='ignore',invalid='ignore')
import itertools
import random
import sys
import time
from collections import Counter
import matplotlib.pyplot as plt
import csv 
import cPickle as pickle
np.set_printoptions(threshold=np.inf)
from sklearn import decomposition
from sklearn import svm
from sklearn import metrics
from sklearn import preprocessing
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC

f1 = open("data", "r")
forward = pickle.load(f1)
f1.close()
train_data = list(forward)[0]
print train_data.shape
train_labels = list(forward)[1]
test_data = np.load('test/test.npy')
pca = decomposition.PCA(n_components=50, whiten=True)
pca.fit(train_data)
X_train_pp = pca.transform(train_data)
X_test_pp = pca.transform(test_data)
# X_train_pp = preprocessing.scale(X_train_pca)
# X_test_pp = preprocessing.scale(X_test_pca)

# X_train_pp = (X_train_pp - X_train_pp.mean(axis=0)) / (X_train_pp.std(axis=0))
# X_test_pp = (X_test_pp - X_test_pp.mean(axis=0)) / (X_test_pp.std(axis=0))

m,n = X_train_pp.shape
for i in range(0,m):
	min_i = min(X_train_pp[i])
	X_train_pp[i] = X_train_pp[i]-min_i
	max_i = max(X_train_pp[i])	
	X_train_pp[i] = X_train_pp[i]/max_i

m1,n1 = X_test_pp.shape
for i in range(0,m1):
	min_i = min(X_test_pp[i])
	X_test_pp[i] = X_test_pp[i]-min_i
	max_i = max(X_test_pp[i])	
	X_test_pp[i] = X_test_pp[i]/max_i
	
Y_train = np.array(list(train_labels.items()))
Y_train = Y_train[:,1]
clf = svm.SVC(kernel='linear', C=1)
# clf = svm.SVC(decision_function_shape='ovo')
clf.fit(X_train_pp, Y_train)
print "training done"
# clf = OneVsOneClassifier(LinearSVC(C=1)).fit(X_train_pp, Y_train)
predict_labels_train = clf.predict(X_train_pp) 
print "train prediction done"
predict_labels_test = clf.predict(X_test_pp)
print predict_labels_test
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

with open('test_b.csv', 'w') as outfile:
	writer = csv.writer(outfile)
	writer.writerow(['ID', 'CATEGORY'])
	for i in range(0,len(predict_labels_test)):
		lab = predict_labels_test[i]
		writer.writerow([str(i),labels_class[lab]])
