close all;
clear all;
clc;
X1= load('q4x.dat');
fileId = fopen('q4y.dat');
text = textscan(fileId,'%s');
fclose(fileId);
Y = text{1};
[m,m1] = size(X1);
disp(m);
disp(m1);
[n,n1] = size(Y);

mu = zeros([1,m1]);
sigma = zeros([1,m1]);

for j = 1:m1
	mu(1,j) = mean(X1(:,j));
	sigma(1,j) = std(X1(:,j));
    for i=1:m
        X1(i,j) = (X1(i,j)-mu(1,j))/sigma(1,j);
    end
end
        
n_alaska = 0;
n_canada = 0;
cntA = 0;
cntC = 0;
A = [];
C = [];
%%%%%%%%%%%% PART a %%%%%%%%%%%%%%%%
mu1 = zeros(1,m1);
mu0 = zeros(1,m1);
for i = 1:n
    if strcmp(Y(i),'Alaska')
        A(cntA+1) = i;
        cntA = cntA + 1;
        for j = 1:m1
            mu0(1,j) = mu0(1,j) + X1(i,j);
        end
    else
        C(cntC+1) = i;
        cntC = cntC+1;
        for j = 1:m1
            mu1(1,j) = mu1(1,j) + X1(i,j);
        end
    end
end
phi = cntC/m;
sum0 = 0;
sum1 = 0;
disp(n);
disp(A);
disp(C);
u0 = mu0/cntC;
disp('u0');
disp(u0);

u1 = zeros([1,m1]);
u1 = mu1/cntC;
disp('u1');
disp(u1);
sum0 = 0;
sum1 = 0;
disp(size(X1(1,:)));
for i = 1:cntA
    inx = A(i);
    sum0 = sum0 + ((X1(inx,:)-u0)'*(X1(inx,:)-u0));
end
sigma0 = sum0/cntA;
disp('sigma0');
disp(sigma0);
for i = 1:cntC
    inx = C(i);
    sum1 = sum1 + ((X1(inx,:)-u1)'*(X1(inx,:)-u1));
end
sigma1 = sum1/cntC;
sigma = (sum0+sum1)/m;
disp('sigma');
disp(sigma);
disp('sigma1');
disp(sigma1);

%%%%%%%%%%% PART b %%%%%%%%%%%%
for i = 1:n
    if strcmp(Y(i),'Alaska')
        plot(X1(i,1) , X1(i,2), ['+' 'r']);
        hold on;
    else
        plot(X1(i,1) , X1(i,2), ['*' 'b']);
        hold on;
    end
end
title('+: Alaska , *: Canada');
xlabel('x1');
ylabel('x2');

%%%%%%%%% PART c %%%%%%%%%%%%
u1 = u1';
u0 = u0';
size((u1'*inv(sigma)*u1)-(u0'*inv(sigma)*u0))

disp('phi');
disp((u1'*inv(sigma)*u1)-(u0'*inv(sigma)*u0));
constant = log((1-phi)/phi)+0.5*((u1'*inv(sigma)*u1)-(u0'*inv(sigma)*u0));
size(X1)
A = (u0'-u1')*inv(sigma);
theta = [A constant];
disp('constant');
disp(constant);
disp('A');
disp(A);
plot(X1(:,1), (((((-1)*A(1,1))/A(1,2))*X1(:,1))-(constant/A(1,2))),'LineWidth',1);

%%%%%%%%%%% PART e %%%%%%%%%%

u1 = u1';
u0 = u0';
P = 0.5 * (inv(sigma0) - inv(sigma1))
Q = (u1*inv(sigma1))-(u0*inv(sigma0))
a = P(1,1)
b = 2*P(1,2)
c = P(2,2)
d = Q(1,1)
e = Q(1,2)
f = 0.5*(log(det(sigma0)/det(sigma1))) + log(phi/(1-phi)) - 0.5*((u1*inv(sigma1))*u1') + 0.5*((u0*inv(sigma0))*u0');
syms x1 x2
ezplot((a*(x1^2) + b*x1*x2 + c*(x2^2) + d*x1 + e*x2 + f), [-3,4,-3,4]);
title('line: sigma1=sigma2 & curve: sigma1 != sigma2');