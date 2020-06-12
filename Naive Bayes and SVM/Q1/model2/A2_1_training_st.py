import sys
import math
import string
from sets import Set
from random import randint
import time
import pickle

def writefile(out_file,Y):
	with open(out_file, 'w') as out:
		for i in xrange(len(Y)):
			out.write(str(Y[i]))
			if (i<len(Y)-1):
				out.write('\n')

f = open("Q1/model2/model1_2", "r")
forward = pickle.load(f)
f.close()
vocabulary = list(forward)[0]
count_class  = list(forward)[1]
class_word_map  = list(forward)[2]
numExamples  = list(forward)[3]
all_labels  = list(forward)[4]

# test_labels =  [line.rstrip('\n') for line in open('imdb_train_labels.txt')]
test_lines = [line.rstrip('\n') for line in open(sys.argv[1])]
ConfusionMatrix = {}

for i in range(1,9):
	ConfusionMatrix[i] = {}
	for j in range(1,9):
		(ConfusionMatrix[i])[j] = 0

outY = {}
theta_sum = {}
total_count = 0.0
correct_count = 0.0
for i in range(0, len(test_lines)):
	total_count += 1
	# actual_label = int(test_labels[i])
	# if actual_label > 4:
	# 	actual_label -= 2
	line = test_lines[i]
	max_value = -sys.maxint
	splitline = line.split()
	for j in range(1,9):
		theta_sum[j] = math.log(count_class[j]/numExamples)
		total = sum(class_word_map[j].values())+len(vocabulary)
		for k in range(0,len(splitline)):
			word = splitline[k]
			word = (word.translate(None, string.punctuation)).lower()
			if ((class_word_map[j]).get(word) == None):
				theta_sum[j] += math.log(1.0/total)
			else:
				theta_sum[j] += math.log(((class_word_map[j])[word]+1)/total)
		if (theta_sum[j] > max_value):
			max_value = theta_sum[j]
			computed_label = j

	# (ConfusionMatrix[actual_label])[computed_label] += 1	
	# if (computed_label == actual_label):
	# 	correct_count += 1
	if computed_label > 4:
		computed_label += 2
	outY[i] = computed_label

writefile(sys.argv[2], outY)

# print "Accuracy : ",(correct_count/total_count)*100,"%\n"
# print "Confusion Matrix"
# print ConfusionMatrix

# total_count_rand = 0.0
# correct_count_rand = 0.0
# for i in range(0,len(test_labels)):
# 	total_count_rand += 1
# 	actual_label_rand = int(test_labels[i])
# 	rand = randint(0,7)
# 	all_labels = list(all_labels)
# 	computed_label_rand = all_labels[rand]

# 	if (computed_label_rand == actual_label_rand):
# 		correct_count_rand += 1

# print total_count_rand
# print correct_count_rand
# print "Accuracy : ",((correct_count_rand*100)/total_count_rand),"%\n"

# max_val = -sys.maxint
# computed_label_max = 0
# total_count_max = 0.0
# correct_count_max = 0.0

# for label in range(1,9):
# 	if (count_class[label] > max_val):
# 		max_val = count_class[label]
# 		computed_label_max = label

# for i in range(0,len(test_labels)):
# 	total_count_max += 1
# 	actual_label_max = int(test_labels[i])
# 	if actual_label_max > 4:
# 		actual_label_max -= 2
# 	if (computed_label_max == actual_label_max):
# 		correct_count_max += 1

# print total_count_max
# print correct_count_max
# print "Accuracy : ",(correct_count_max/total_count_max)*100,"%\n"