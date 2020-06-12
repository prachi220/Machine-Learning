import pickle
import numpy as np
from numpy import exp, array, random, dot
import matplotlib.pyplot as plt

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
f1 = open("neural_nets_mnist", "r")
forward = pickle.load(f1)
f1.close()
layers1 = list(forward)[0]
X1 = np.loadtxt("mnist_data/MNIST_train.csv", dtype=float, delimiter=',')
m,n = X1.shape
X = np.array(X1[:,0:n-1])
Y = np.array(X1[:,n-1])
Y1 = change_data(X,Y,m)
Xt1 = np.loadtxt("mnist_data/MNIST_test.csv", dtype=float, delimiter=',')
m1,n1 = Xt1.shape
Xt = np.array(Xt1[:,0:n1-1])
Yt = np.array(Xt1[:,n1-1])
Yt1 = change_data(Xt,Yt,m1)

print "train_data ", X
print "train_label ", Y1

def sigmoid(x):
    e1 = 1 + exp(-x)
    e2 = 1/e1
    return e2

def test(test_data, layers1):
    layers1_output = {}
    layers1_output[0] = sigmoid(dot(test_data, layers1[0]))
    for k in range(1,len(layers1)):
        layers1_output[k] = sigmoid(dot(layers1_output[k-1], layers1[k]))
    return layers1_output

def predict(t):
    print "t : ", t
    y_predict = []
    for i in xrange(0,len(t)):
        test_output = test(t[i],layers1)
        if test_output[len(layers1)-1] > 0.5:
            y_predict.append(1)
        else:
            y_predict.append(0)
    return np.asarray(y_predict)


def get_accuracy(output, label):
	accuracy = 0.0
	if output[len(layers1)-1] > 0.5:
		if label==1:
			accuracy = 1
	else:
		if label==0:
			accuracy = 1
	return accuracy

accuracy = 0.0
data = Xt
label = Yt1
for i in xrange(0,len(data)):
    test_output = test(data[i],layers1)
    print test_output[len(layers1)-1],label[i]
    accuracy += get_accuracy(test_output,label[i])
print accuracy/len(label)

# def plot_decision_boundary(model, X, y):
#     """
#     Given a model(a function) and a set of points(X), corresponding labels(y), scatter the points in X with color coding
#     according to y. Also use the model to predict the label at grid points to get the region for each label, and thus the 
#     descion boundary.
#     Example usage:
#     say we have a function predict(x,other params) which makes 0/1 prediction for point x and we want to plot
#     train set then call as:
#     plot_decision_boundary(lambda x:predict(x,other params),X_train,Y_train)
#     params(3): 
#     model : a function which expectes the point to make 0/1 label prediction
#         X : a (mx2) numpy array with the points
#         y : a (mx1) numpy array with labels
#     outputs(None)
#     """
#     print "shape1 ", X.shape
#     print "shape2 ", y.shape
#     # Set min and max values and give it some padding
#     x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
#     y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
#     h = 0.02
#     # Generate a grid of points with distance h between them
#     xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
#     # Predict the function value for the whole grid
#     Z = model(np.c_[xx.ravel(), yy.ravel()])
#     Z = Z.reshape(xx.shape)
#     # Plot the contour and training examples
#     plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
#     plt.ylabel('x2')
#     plt.xlabel('x1')
#     plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
#     plt.show()

# plot_decision_boundary(predict,data,label)