close all;
clear all;
clc;
X1= csvread('linearX.csv');
Y= csvread('linearY.csv');
[m,m1] = size(X1);
X1 = normalise(X1);       
X = [ones(m,1) X1];
notconverged = true;
theta = [0; 0];
theta_new = [0; 0];
disp(theta);
disp(size(theta));
disp(size(Y));
n = 0.001;
J_theta_prev = 0;
for i = 1:m
    J_theta_prev = J_theta_prev + (Y(i))^2;
end
J_theta_prev = J_theta_prev/(2);
figure;
theta2 = -0.002:0.001:0.007
theta1 = -0.5:0.25:1.75
[xx,yy] = meshgrid(theta1,theta2);          %creating the meshgrid
J_theta  = zeros(length(theta1),length(theta2));
for a = 1: length(theta1)
    for b = 1: length(theta2)
        sum =0 ;
        for y= 1:m
            sum = sum + ((Y(y) - (theta1(a) +theta2(b)*X1(y)))^2);
        end
        J_theta(a,b) = sum/(2);
     end
end
mesh(xx,yy,J_theta); 
xlabel('theta0');
ylabel('theta1');
zlabel('J(theta)');
rotate3d
hold on; 

% figure;
% theta2 = -4:2:10;
% theta1 = -4:2:10;
% [xx,yy] = meshgrid(theta1,theta2);          %creating the meshgrid
% J_theta  = zeros(length(theta1),length(theta2));
% for a = 1: length(theta1)
%     for b = 1: length(theta2)
%         sum =0 ;
%         for y= 1:m
%             sum = sum + ((Y(y) - (theta1(a) +theta2(b)*X1(y)))^2);
%         end
%         J_theta(a,b) = sum/(2);
%      end
% end
% contour(xx,yy,J_theta);
% hold on; 
% xlabel('theta0');
% ylabel('theta1');
% zlabel('J(theta)');

while notconverged
%     scatter(theta(1),theta(2), J_theta_prev);
    scatter3(theta(1),theta(2), J_theta_prev);
    hold on; 
    t1 = 0;
    t2 = 0;
    for i=1:m
        t1 = t1 + (theta(1) + X1(i)*theta(2)-Y(i));
        t2 = t2 + (theta(1) + X1(i)*theta(2)-Y(i))*X1(i);
    end
    theta_new(1) = theta(1) - n*t1;   
    theta_new(2) = theta(2) - n*t2;
%     disp(J_theta_prev);
    J_theta_curr=0;
    for i=1:m
        J_theta_curr = J_theta_curr + ((Y(i)-(dot(theta, X(i,:))))^2);
    end
    J_theta_curr = J_theta_curr/(2)
    %check convergence condition
    
%      if abs(J_theta_curr - J_theta_prev) < 0.000000000001
%         notconverged = false;
%      else
%          J_theta_prev = J_theta_curr;
%      end
    if norm(theta_new-theta) < 0.00000001 && abs(J_theta_curr - J_theta_prev) < 0.000000000001
        notconverged = false;
    else
        theta = theta_new;
        J_theta_prev = J_theta_curr;
    end
%     scatter3(theta(1),theta(2), J_theta_curr);    %plotting the value of each J(theta) corresponding to the parameters obtained 
     pause(0.02);
end
disp(theta);
x=-2:5;
figure,
plot(X1, Y, ['.' 'r']);
hold on;
plot(x,theta(2)*x+theta(1));
hold off;
xlabel('x');
ylabel('y');


