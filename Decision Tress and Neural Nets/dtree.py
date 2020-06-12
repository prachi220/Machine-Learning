import sys
import math
from sets import Set
from random import randint
import string
import pickle

num_leaf=0


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

def find_best_split(data):
	best_gain = 0 
	best_attr = None
	current_entropy = get_entropy(data)
	num_attributes= len(data[0])
	# print "current_entropy" , current_entropy
	for col in range(1, num_attributes):
		attr = col
		values = get_attribute_values(data,attr)
		children = dict()
		children_entropy = dict()
		total_len = 0.0
		divide = 1
		for attr_val in values:
			parts = get_true_rows(data, attr, attr_val)
			children[attr_val]= parts
			child_entr = get_entropy(parts)
			children_entropy[attr_val]= child_entr
			total_len += len(parts)
			# if len(parts) == 0:
	 	# 		divide = 0
	 	# if not divide:
	 	# 	continue
	 	subtract = 0.0
	 	for i in values:
	 		wt = len(children[i])/total_len
			# print children_entropy[i] 
			subtract += wt*children_entropy[i]
		# print "subtract", subtract
		info_gain = current_entropy - subtract
	 	if info_gain > best_gain:
	 		best_gain= info_gain
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
		return Node(1, max_cls, None, None, None, 0)
	gain, attribute = find_best_split(data)
	if gain == 0:
		return Node(1, max_cls, None, None, None, 0)
	values = get_attribute_values(data,attribute)
	original_values = attribute_values[attribute]
	children = dict()
	for attr_val in values:
		child = get_true_rows(data, attribute, attr_val)
		children[attr_val] = build_tree(child)
	return Node(0, max_cls, attribute, values, children, 0)


def classify(row, node):
	if  node.isLeaf == 1:
		return node.cls
	else:
		attr = node.attribute
		attr_values = node.attribute_values
		children = node.children
		for i in attr_values:
			if row[attr] == i:
				return classify(row, children[i])


def count_unmarked_nodes(root):
	if root.isLeaf==1:
		if root.marked==0:
			return 1
		else:
			return 0
	else:
		attr_values = root.attribute_values
		children = root.children
		total = 0
		for i in attr_values:
			total += count_unmarked_nodes(children[i])
		if root.marked==0:
			return total+1
		else:
			return total


def mark_all(root):
	if root.isLeaf == 1:
		# print root.marked
		root.marked = 1
		# print root.marked
		return;
	else:
		# print root.marked
		root.marked = 1
		# print root.marked
		attr_values = root.attribute_values
		children = root.children
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
			for i in attr_values:
				if row[attr] == i:
					return classify_unmarked(row, children[i])

def unmark_node(node):
	if node.marked == 1:
		node.marked = 0
		return;
	else:
		if node.isLeaf==1:
			return;
		else:
			attr_values = node.attribute_values
			children = node.children
			for i in attr_values:
				unmark_node(children[i])

def walk_tree(root, data):
	for i in range(0,12):
		accuracy = 0.0
		for row in data:
			cls = classify_unmarked(row, my_tree)
			if row[0]== cls:
				accuracy += 1.0
		unmark_node(root)
		n = count_unmarked_nodes(my_tree)
		print n, (accuracy*100)/len(data)
		nodes_accuracy[n] = (accuracy*100)/len(data)



if __name__ == '__main__':
	f = open("dtree_data", "r")
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

	accuracies = {}
	my_tree = build_tree(train_data)
	accuracy = 0.0
	for row in data:
		if row[0]== classify(row, my_tree):
			accuracy += 1.0
	print accuracy/len(data);

	num_nodes = 0
	nodes_accuracy = {}
	nodes_accuracy = {}

	# for i in range(0,1000):

	# mark_all(my_tree)
	# print "cnt, ", count_unmarked_nodes(my_tree)
	# mark_all(my_tree)
	# print "marked"
	# print "cnt, ", count_unmarked_nodes(my_tree)
	# # for i in xrange(1,15):
	# # 	unmark_node(my_tree)
	# # 	print "cnt, ", count_unmarked_nodes(my_tree)
	# walk_tree(my_tree, data)

	# print "cls " , my_tree.cls
	# for row in data:
	# 	print classify(row, my_tree)

