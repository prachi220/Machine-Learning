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
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import numpy
# fix random seed for reproducibility
numpy.random.seed(7)
# load pima indians dataset
f1 = open("data", "r")
forward = pickle.load(f1)
f1.close()
X = list(forward)[0]
train_labels = list(forward)[1]
Y_train = np.array(list(train_labels.items()))
Y = Y_train[:,1]
print Y.shape
test_data = np.load('test/test.npy')

# m,n = X.shape
# for i in range(0,m):
# 	min_i = min(X[i])
# 	X[i] = X[i]-min_i
# 	max_i = max(X[i])	
# 	X[i] = X[i]/max_i

# m1,n1 = test_data.shape
# for i in range(0,m1):
# 	min_t = min(test_data[i])
# 	test_data[i] = test_data[i]-min_t
# 	max_t = max(test_data[i])	
# 	test_data[i] = test_data[i]/max_t

X = X/255.0
test_data = test_data/255.0
# X = (X - X.mean(axis=0)) / (X.std(axis=0))
# test_data = (test_data - test_data.mean(axis=0)) / (test_data.std(axis=0))

# create model
model = Sequential()
model.add(Dense(1000, activation='sigmoid', input_dim=X.shape[1]))
model.add(Dense(20, activation='softmax'))
# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
# Convert labels to categorical one-hot encoding
one_hot_labels = to_categorical(Y, num_classes=20)
model.fit(X, one_hot_labels, epochs=15, batch_size=300)
# evaluate the model
# scores = model.evaluate(X, one_hot_labels)
# print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

out_test = model.predict(test_data)
predict_labels_test = np.argmax(out_test, axis=1)
print predict_labels_test

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
with open('test_c.csv', 'w') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(['ID', 'CATEGORY'])
    for i in range(0,len(predict_labels_test)):
    	lab = predict_labels_test[i]
    	writer.writerow([str(i),labels_class[lab]])

