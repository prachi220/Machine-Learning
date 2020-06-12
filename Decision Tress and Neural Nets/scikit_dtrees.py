from sklearn import tree
import sys
import pickle

f = open("dtree_data", "r")
forward = pickle.load(f)
f.close()
train_data = list(forward)[0]
test_data  = list(forward)[1]
valid_data  = list(forward)[2]
train_labels = []
train_attributes = []
for i in xrange(0,len(train_data)):
	train_labels.append(train_data[i][0])
	train_attributes.append(train_data[i][1:])

max_valid = 0
max_param = []
for split in range(2,10):
	for leaf in range(2,10):
		for depth in range(5,15):
			print split,leaf,depth
			clf = tree.DecisionTreeClassifier(min_samples_split=split,min_samples_leaf=leaf,max_depth=depth)
			clf = clf.fit(train_attributes, train_labels)

			data = valid_data
			labels = []
			test_attributes = []
			for i in xrange(0,len(data)):
				labels.append(data[i][0])
				test_attributes.append(data[i][1:])

			predict_labels = clf.predict(test_attributes)
			correctCount_v = 0.0
			for i in range(len(labels)):
				if (predict_labels[i]==labels[i]):
					correctCount_v += 1
			print "valid ", correctCount_v/len(test_attributes)
			if (correctCount_v/len(test_attributes)) > max_valid:
				max_valid = correctCount_v/len(test_attributes)

			data = test_data
			labels = []
			test_attributes = []
			for i in xrange(0,len(data)):
				labels.append(data[i][0])
				test_attributes.append(data[i][1:])

			predict_labels = clf.predict(test_attributes)
			correctCount = 0.0
			for i in range(len(labels)):
				if (predict_labels[i]==labels[i]):
					correctCount += 1

			print "test ", correctCount/len(test_attributes)

			data = train_data
			labels = []
			test_attributes = []
			for i in xrange(0,len(data)):
				labels.append(data[i][0])
				test_attributes.append(data[i][1:])

			predict_labels = clf.predict(test_attributes)
			correctCount = 0.0
			for i in range(len(labels)):
				if (predict_labels[i]==labels[i]):
					correctCount += 1

			print "train ", correctCount/len(test_attributes)
print "max ", max_valid
