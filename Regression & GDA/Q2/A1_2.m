close all;
clear all;
clc;
X1= csvread('weightedX.csv');
Y= csvread('weightedY.csv');
[m,m1] = size(X1);

X1= normalise(X1);
disp(X1);
X = [ones(m,1) X1];
disp(X);
t1 = X'*X;
theta = t1\(X'*Y);
size(theta)
    size(X)
fn = X*theta;
figure,
plot(X1, Y, ['.' 'r']);
hold on;
plot(X1,fn);
hold on;

t = 0.1;
W = zeros([m,m]);
h_theta= zeros([m,1]);
for i = 1:m
    for j = 1:m
        n = (X1(i)-X1(j))^2;
        d = 2*(t^2);
        W(j,j) = exp((-1)*n/d);
    end
    t1 = X'*W*X;
    theta1 = t1\(X'*W*Y);
    h_theta(i) = dot(X(i,:),theta1);
end
plot(X1, h_theta,['.' 'b'],'LineWidth',1); 
hold off;
xlabel('x');
ylabel('y');
title('locally weighted, t=10');
