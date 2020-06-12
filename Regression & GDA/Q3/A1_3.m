close all;
clear all;
clc;
X1= csvread('logisticX.csv');
Y= csvread('logisticY.csv');
[m,m1] = size(X1);

mu = zeros([1,m1]);
sigma = zeros([1,m1]);

for j = 1:m1
	mu(1,j) = mean(X1(:,j));
	sigma(1,j) = std(X1(:,j));
    for i=1:m
        X1(i,j) = (X1(i,j)-mu(1,j))/sigma(1,j);
    end
end

X = [ones(m,1) X1];
n = m1+1;
theta_prev = zeros([n,1]);
notconverged = true;
while(notconverged)
    der = zeros([n,1]);
    H = zeros([n,n]);
    for i = 1:n
        for j = 1:n
            for k = 1:m
                H(i,j) = H(i,j) + sigmoid(dot(theta_prev,X(k,:)))*(1-sigmoid(dot(theta_prev,X(k,:))))*X(k,i)*X(k,j);
            end
        end
    end
    for i = 1:n
        for k = 1:m
            der(i) = der(i) + (Y(k)-sigmoid(dot(theta_prev,X(k,:))))*X(k,i);
        end
    end
    theta_new = theta_prev - H\der
    theta_new = theta_new/norm(theta_new)
    disp('norm');
    disp(norm(theta_new-theta_prev,inf));
    
    if norm((theta_new-theta_prev),inf) < 0.000001
        notconverged = false;
    else
        theta_prev = theta_new;
    end
end
theta = theta_prev;
disp('theta');
disp(theta);

for i =1:m                         %ploting the dataset 
    if(Y(i) == 1)
        plot(X1(i,1) , X1(i,2), ['+' 'r']);
        hold on;
    end
    if (Y(i)==0)
        plot(X1(i,1) , X1(i,2), ['*' 'b']);
        hold on;  
    end        
end

P1 = X1(:,1);
plot_y = (-1./theta(3))*(theta(2).*P1 + theta(1));
plot(P1, plot_y);
hold off;
title('+: y=1 , *: y=0');
xlabel('x1');
ylabel('x2');
    

         