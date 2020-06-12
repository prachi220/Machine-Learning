import numpy as np
import sys
import math
from sets import Set
from random import randint
import string
import pickle

class_word_map = dict()
count_class = dict()
vocabulary = set()
numExamples = 0;
labels = [line.rstrip('\n') for line in open('imdb_train_labels.txt')]
lines = [line.rstrip('\n') for line in open('imdb_train_text_stemstop.txt')]
remove_words = {'movi', 'film', 'very', 'guy', 'girl', 'think', 'enough', 'men', 'women'}
good_words = {'good', 'epic', 'best', 'brilliant', 'amaz', 'excellent', 'recommend', 'oscar'}
good_words_pairs = {('must', 'see'), ('must', 'watch'),('love', 'movi'), ('love' , 'film'), ('strong','movi'), ('strong', 'film')}
count = 0
counter = 0
all_labels = set()

for i in range(0,len(labels)):
	numExamples += 1
	lab = int(labels[i])
	if lab > 4:
		lab -= 2
	if class_word_map.get(lab) is None:
		all_labels.add(lab)
		class_word_map[lab] = {}
		count_class[lab] = 0.0
	count_class[lab] = count_class[lab]+1
	line = lines[i]
	splitLine = line.split()
	for j in range(0, len(splitLine)-1):
		# print j
		word1 = splitLine[j]
		word2 = splitLine[j+1]
		if (word1,word2) not in vocabulary:
		    vocabulary.add((word1,word2))
		if ((class_word_map[lab]).get((word1,word2))) is None:
		    (class_word_map[lab])[(word1,word2)] = 0.0
		# if word in good_words:
		# 	(class_word_map[lab])[(word] += 10
		# else:
		(class_word_map[lab])[(word1,word2)] += 1


f = open("model1_3", "w")
pickle.dump([vocabulary, count_class, class_word_map, numExamples, all_labels], f)
f.close()


