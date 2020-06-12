from sklearn.ensemble import RandomForestClassifier
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
max_cnt = 0
for est in range(1,15):
		for fet in range(2,11):
			print est,fet
			clf = RandomForestClassifier(n_estimators=est,max_features=fet,bootstrap=True)
			clf = clf.fit(train_attributes, train_labels)

			data = valid_data
			test_labels = []
			test_attributes = []
			for i in xrange(0,len(data)):
				test_labels.append(data[i][0])
				test_attributes.append(data[i][1:])

			predict_labels = clf.predict(test_attributes)

			correctCount_v = 0.0
			for i in range(len(test_labels)):
				if (predict_labels[i]==test_labels[i]):
					correctCount_v += 1

			print float(correctCount_v)/len(test_labels)
			if float(correctCount_v)/len(test_labels) > max_cnt:
				max_cnt = float(correctCount_v)/len(test_labels)

			data = test_data
			test_labels = []
			test_attributes = []
			for i in xrange(0,len(data)):
				test_labels.append(data[i][0])
				test_attributes.append(data[i][1:])

			predict_labels = clf.predict(test_attributes)

			correctCount = 0.0
			for i in range(len(test_labels)):
				if (predict_labels[i]==test_labels[i]):
					correctCount += 1

			print float(correctCount)/len(test_labels)

			data = train_data
			test_labels = []
			test_attributes = []
			for i in xrange(0,len(data)):
				test_labels.append(data[i][0])
				test_attributes.append(data[i][1:])

			predict_labels = clf.predict(test_attributes)

			correctCount = 0.0
			for i in range(len(test_labels)):
				if (predict_labels[i]==test_labels[i]):
					correctCount += 1

			print float(correctCount)/len(test_labels)

print "max_cnt, ", max_cnt