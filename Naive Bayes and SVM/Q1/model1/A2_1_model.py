import numpy as np
import sys
import math
from sets import Set
from random import randint
import string
import pickle

class_word_map = {}
count_class = {}
vocabulary = set()
numExamples = 0;
labels = [line.rstrip('\n') for line in open('imdb_train_labels.txt')]
lines = [line.rstrip('\n') for line in open('imdb_train_text.txt')]
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
	for word in splitLine:
		word = (word.translate(None, string.punctuation)).lower()
		if word not in vocabulary:
		    vocabulary.add(word)
		if ((class_word_map[lab]).get(word)) is None:
		    (class_word_map[lab])[word] = 0.0
		(class_word_map[lab])[word] += 1  

f = open("model1_1", "w")
pickle.dump([vocabulary, count_class, class_word_map, numExamples, all_labels], f)
f.close()


