import pickle
import numpy as np
from numpy import exp, array, random, dot
import matplotlib.pyplot as plt

print "Stage 3) Considering a new situation"
f = open("neural_nets", "r")
forward = pickle.load(f)
f.close()
layers = list(forward)[0]
train_data = list(forward)[1]
train_label = list(forward)[2]
test_data = list(forward)[3]
test_label = list(forward)[4]
num_inputs = len(train_data)
train_data = np.loadtxt("toy_data/toy_trainX.csv", dtype=float, delimiter=',')
train_label = np.loadtxt("toy_data/toy_trainY.csv", dtype=float, delimiter=',')

test_data = np.loadtxt("toy_data/toy_testX.csv", dtype=float, delimiter=',')
test_label = np.loadtxt("toy_data/toy_testY.csv", dtype=float, delimiter=',')

def sigmoid(x):
    e1 = 1 + exp(-x)
    e2 = 1/e1
    return e2

def test(test_data, layers):
    layers_output = {}
    layers_output[0] = sigmoid(dot(test_data, layers[0]))
    for k in range(1,len(layers)):
        layers_output[k] = sigmoid(dot(layers_output[k-1], layers[k]))
    return layers_output

def predict(t):
    y_predict = []
    for i in xrange(0,len(t)):
        test_output = test(t[i],layers)
        if test_output[len(layers)-1] > 0.5:
            y_predict.append(1)
        else:
            y_predict.append(0)
    return np.asarray(y_predict)


def get_accuracy(output, label):
	accuracy = 0.0
	if output[len(layers)-1] > 0.5:
		if label[i]==1:
			accuracy = 1
	else:
		if label[i]==0:
			accuracy = 1
	return accuracy

accuracy = 0.0
data = test_data
label = test_label
for i in xrange(0,len(data)):
    test_output = test(data[i],layers)
    print test_output[len(layers)-1],label[i]
    accuracy += get_accuracy(test_output,label)
print accuracy/len(label)

def plot_decision_boundary(model, X, y):
    """
    Given a model(a function) and a set of points(X), corresponding labels(y), scatter the points in X with color coding
    according to y. Also use the model to predict the label at grid points to get the region for each label, and thus the 
    descion boundary.
    Example usage:
    say we have a function predict(x,other params) which makes 0/1 prediction for point x and we want to plot
    train set then call as:
    plot_decision_boundary(lambda x:predict(x,other params),X_train,Y_train)
    params(3): 
    model : a function which expectes the point to make 0/1 label prediction
        X : a (mx2) numpy array with the points
        y : a (mx1) numpy array with labels
    outputs(None)
    """
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = 0.02
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.show()

plot_decision_boundary(predict,data,label)