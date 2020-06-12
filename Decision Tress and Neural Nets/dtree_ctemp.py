import sys
import math
from sets import Set
from random import randint
import string
import pickle
import statistics
import numpy as np

num_leaf=0
num_node=0
non_cnt = {2,4,6,7,8,9,10,14}

class Node:
	def __init__(self, isLeaf, cls, attribute, attribute_values, children, marked):
		self.isLeaf = isLeaf
		self.cls = cls
		self.attribute = attribute
		self.attribute_values = attribute_values
		self.children = children
		self.marked = marked


def get_true_rows(data, attr, attr_val):
	true_rows = []
	for row in data:
		row_val = row[attr]
		if row_val== attr_val:
			true_rows.append(row)
	return true_rows

def get_entropy(data):
	probability = {}
	probability[0] = 0.0
	probability[1] = 0.0
	total = 0.0
	entropy = 0.0
	for row in data:
		label = row[0]
		probability[label] += 1 
		total += 1
	entropy=0.0
	# print "entropy called"
	for i in range(0,len(probability)):
		r = probability[i]
		p=r/total
		if ( p == 0):
			return 1.0
		else:
			entropy -= p*(math.log(p,2))
	return entropy

def get_attribute_values(data,attr):
	values = set()
	for row in data:
		v = row[attr]
		if v not in values:
			values.add(v)
	return values

def get_attribute_values_med(data,attr):
	values = []
	for row in data:
		v = row[attr]
		values.append(v)
	return values

def find_best_split(data):
	best_gain = 0 
	best_attr = None
	current_entropy = get_entropy(data)
	num_attributes= len(data[0])
	# print "current_entropy" , current_entropy
	for col in range(1, num_attributes):
		attr = col
		children = dict()
		children_entropy = dict()
		total_len = 0.0
		divide = 1
		if attr in non_cnt:
			values = get_attribute_values(data,attr)
			for attr_val in values:
				parts = get_true_rows(data, attr, attr_val)
				children[attr_val]= parts
				child_entr = get_entropy(parts)
				children_entropy[attr_val]= child_entr
				total_len += len(parts)
		else:
			values = get_attribute_values_med(data,attr)
			# print "values: ", values
			med = statistics.median(values)
			# print "median: ", med
			values = set(values)
			child1 = []
			child2 = []
			for attr_val in values:
				if attr_val <= med:
					child1 += get_true_rows(data, attr, attr_val)
				else:
					child2 += get_true_rows(data, attr, attr_val)
			children[0] = child1
			children[1] = child2
			if len(child1) == 0:
				children_entropy[0] = 0
				# continue
			else:
				child1_entr = get_entropy(child1)
				children_entropy[0]= child1_entr
			if len(child2) == 0:
				children_entropy[1] = 0
				# continue
			else:
				child2_entr = get_entropy(child2)
				children_entropy[1]= child2_entr
			total_len = len(child1)+len(child2)
			values = set([0,1])
	 	subtract = 0.0
	 	for i in values:
	 		wt = len(children[i])/total_len
			# print children_entropy[i] 
			subtract += wt*children_entropy[i]
		# print "subtract", subtract
		info_gain = current_entropy - subtract
	 	if info_gain >= best_gain:
	 		best_gain = info_gain
	 		best_attr = attr
	return best_gain, best_attr


def build_tree(data):
	counts = dict()
	counts[0] = 0
	counts[1] = 0
	for row in data:
		label = row[0]
		if label == 1:
			counts[1] += 1
		else:
			counts[0] += 1
	if counts[1] > counts[0]:
		max_cls = 1
	else:
		max_cls = 0
	if len(data) == 4:
		return Node(1, max_cls, None, None, None,0)
	gain, attribute = find_best_split(data)
	if gain == 0:
		return Node(1, max_cls, None, None, None,0)
	# values = get_attribute_values(data,attribute)
	original_values = attribute_values[attribute]
	children = dict()
	if attribute in non_cnt:
		values = get_attribute_values(data,attribute)
		for attr_val in values:
			child = get_true_rows(data, attribute, attr_val)
			children[attr_val] = build_tree(child)
		return Node(0, max_cls, attribute, values, children,0)
	else:
		values = get_attribute_values_med(data,attribute)
		med = statistics.median(values)
		values = set(values)
		child1 = []
		child2 = []
		for attr_val in values:
			if attr_val <= med:
				child1 += get_true_rows(data, attribute, attr_val)
			else:
				child2 += get_true_rows(data, attribute, attr_val)
		if len(child1)==0:
			print "child1 is null"
		if len(child2)==0:
			print "child2 is null"
		children[0] = build_tree(child1)
		children[1] = build_tree(child2)
		print "******************ATTRIBUTE************************", attribute
		return Node(0, max_cls, attribute, set([med]), children,0)

def classify(row, node):
	# print "classify"
	if  node.isLeaf == 1:
		return node.cls
	else:
		attr = node.attribute
		attr_values = node.attribute_values
		children = node.children
		if attr in non_cnt:
			for i in attr_values:
				if row[attr] == i:
					return classify(row, children[i])
		else:
			for med in attr_values:
				# print "See I am being repeated"
				if row[attr] <= med:
					return classify(row, children[0])
				else:
					return classify(row, children[1])

def count_unmarked_nodes(root):
	if root.isLeaf==1:
		if root.marked==0:
			return 1
		else:
			return 0
	else:
		attr = root.attribute
		children = root.children
		total = 0
		if attr not in non_cnt:
			attr_values = set([0,1])
		else:
			attr_values = root.attribute_values
		for i in attr_values:
			total += count_unmarked_nodes(children[i])
		if root.marked==0:
			return total+1
		else:
			return total

def mark_all(root):
	if root.isLeaf == 1:
		root.marked = 1
		return;
	else:
		root.marked = 1
		attr = root.attribute
		children = root.children
		if attr not in non_cnt:
			attr_values = set([0,1])
		else:
			attr_values = root.attribute_values
		for i in attr_values:
			mark_all(children[i])

def classify_unmarked(row, node):
	if node.marked == 1:
		return node.cls
	else:
		if node.isLeaf==1:
			return node.cls
		else:
			attr = node.attribute
			attr_values = node.attribute_values
			children = node.children
			if attr in non_cnt:
				for i in attr_values:
					if row[attr] == i:
						return classify_unmarked(row, children[i])
			else:
				for med in attr_values:
					if row[attr] <= med:
						return classify_unmarked(row, children[0])
					else:
						return classify_unmarked(row, children[1])

def unmark_node(node):
	if node.marked == 1:
		node.marked = 0
		return;
	else:
		if node.isLeaf==1:
			return;
		else:
			attr = node.attribute
			children = node.children
			if attr not in non_cnt:
				attr_values = set([0,1])
			else:
				attr_values = node.attribute_values
			for i in attr_values:
				unmark_node(children[i])

def walk_tree(root, data):
	for i in range(0,30):
		accuracy = 0.0
		for row in data:
			cls = classify_unmarked(row, my_tree)
			if row[0]== cls:
				accuracy += 1.0
		unmark_node(root)
		n = count_unmarked_nodes(my_tree)
		print n, (accuracy*100)/len(data)
		nodes_accuracy[n] = (accuracy*100)/len(data)

def walk_branches(node):
	if  node.isLeaf == 1:
		node.marked = 1
		return;
	else:
		attr = node.attribute
		attr_values = node.attribute_values
		children = node.children
		all_marked = 0
		if attr in non_cnt:
			for i in attr_values:
				if children[i].marked == 0:
					return walk_branches(children[i])
				else:
					all_marked += 1
			if all_marked==len(attr_values):
				node.marked = 1
		else:
			for med in attr_values:
				if cont_attribute_split[attr][med] is None:
					cont_attribute_split[attr][med] = 0
				cont_attribute_split[attr][med] += 1
				for i in range(0,2):
					if children[i].marked == 0:
						return walk_branches(children[i])
					else:
						all_marked += 1
				if all_marked==2:
					node.marked = 1

def walk_branches(node):
	if  node.isLeaf == 1:
		return;
	else:
		attr = node.attribute
		attr_values = node.attribute_values
		children = node.children
		all_marked = 0
		if attr in non_cnt:
			for i in attr_values:
				walk_branches(children[i])
		else:
			for med in attr_values:	
				if cont_attribute_split[attr].get(med) is None:
					# cont_attribute_split[attr].add(med)
					cont_attribute_split[attr][med] = 0
				cont_attribute_split[attr][med] += 1
				for i in range(0,2):
					walk_branches(children[i])	


if __name__ == '__main__':
	f = open("dtree_data_c", "r")
	forward = pickle.load(f)
	f.close()
	train_data = list(forward)[0]
	#print train_data
	test_data  = list(forward)[1]
	valid_data  = list(forward)[2]
	
	data = valid_data
	attribute_values = dict()
	for att in range(1,len(train_data[0])):
		values = get_attribute_values(train_data,att)
		attribute_values[att] = values
	my_tree = build_tree(train_data)
	accuracy = 0.0
	for row in data:
		if row[0]== classify(row, my_tree):
			accuracy += 1.0
	print accuracy/len(data);

	num_nodes = 0
	nodes_accuracy = {}

	# for i in range(0,1000):
	cont_attribute_split = {}
	cont_attribute_split[1] = {}
	cont_attribute_split[3] = {}
	cont_attribute_split[5] = {}
	cont_attribute_split[11] = {}
	cont_attribute_split[12] = {}
	cont_attribute_split[13] = {}
	walk_branches(my_tree)
	print "########### 1",cont_attribute_split[1]
	print "########### 1", cont_attribute_split[3]
	print "########### 1", cont_attribute_split[5]
	print "########### 1", cont_attribute_split[11]
	print "########### 1", cont_attribute_split[12]
	print "########### 1", cont_attribute_split[13]

	# mark_all(my_tree)
	# print "cnt, ", count_unmarked_nodes(my_tree)
	# mark_all(my_tree)
	# print "marked"
	# print "cnt, ", count_unmarked_nodes(my_tree)
	# # for i in xrange(1,50):
	# # 	unmark_node(my_tree)
	# # 	print "cnt, ", count_unmarked_nodes(my_tree)
	# walk_tree(my_tree, data)
	# walk_branches(my_tree)
