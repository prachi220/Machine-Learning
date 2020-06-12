import numpy as np
from sklearn.cluster import KMeans
import csv 
import cPickle as pickle

f1 = open("data", "r")
forward = pickle.load(f1)
f1.close()
X = list(forward)[0]
train_labels = list(forward)[1]
Y_train = np.array(list(train_labels.items()))
train_labels = Y_train[:,1]
print train_labels.shape
X = (X - X.mean(axis=0)) / (X.std(axis=0))
kmeans = KMeans(n_clusters=20, n_init=10)
kmeans.fit(X)
f = open("k_means", "w")
pickle.dump([kmeans,X, train_labels], f)
f.close()

