import pickle
import numpy as np
from numpy import exp, array, random, dot

def change_data(X, Y, m):
    for i in range(0,m):
        min_i = min(X[i])
        X[i] = X[i]-min_i
        max_i = max(X[i])   
        X[i] = X[i]/max_i
        if Y[i]==6:
            Y[i]=0
        else:
            Y[i]=1
    return Y

print "Stage 3) Considering a new situation"
f1 = open("neural_nets_relu", "r")
forward = pickle.load(f1)
f1.close()
layers = list(forward)[0]
X1 = np.loadtxt("mnist_data/MNIST_train.csv", dtype=float, delimiter=',')
m,n = X1.shape
train_data = np.array(X1[:,0:n-1])
Y = np.array(X1[:,n-1])
train_label = change_data(train_data,Y,m)
Xt1 = np.loadtxt("mnist_data/MNIST_test.csv", dtype=float, delimiter=',')
m1,n1 = Xt1.shape
test_data = np.array(Xt1[:,0:n1-1])
Yt = np.array(Xt1[:,n1-1])
test_label = change_data(test_data,Yt,m1)
num_inputs = len(train_data)

def relu(x):
    e2 = []
    for i in range(0,len(x)):
        e2.append(max(0,x[i]))
    e2 = np.asarray(e2)
    # print "e2 herer ", len(e2)
    return e2

def sigmoid(x):
    e1 = 1 + exp(-x)
    e2 = 1/e1
    return e2

def test(test_data, layers):
    layers_output = {}
    layers_output[0] = relu(dot(test_data, layers[0]))
    for k in range(1,len(layers)-1):
        layers_output[k] = relu(dot(layers_output[k-1], layers[k]))
    layers_output[len(layers)-1] = sigmoid(dot(layers_output[len(layers)-2], layers[len(layers)-1]))
    return layers_output

def get_accuracy(output, label):
	accuracy = 0.0
	if output[len(layers)-1] > 0.5:
		if label==1:
			accuracy = 1
	else:
		if label==0:
			accuracy = 1
	return accuracy

accuracy = 0.0
data = test_data
label = test_label
for i in xrange(0,len(data)):
    test_output = test(data[i],layers)
    print test_output[len(layers)-1],label[i]
    accuracy += get_accuracy(test_output,label[i])
print accuracy/len(label)
