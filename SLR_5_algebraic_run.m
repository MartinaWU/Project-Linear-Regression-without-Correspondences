addpath('libs');

%% parameters: 
% NOTE: in this exercise n is limited to be 2,3,4,5.

m = 50000;
n = 3;
sigma = 0;
shuffled_ratio = 0.5;

%% generate data:
[A, y, x] = SLR_1_gen_data(m, n, sigma, shuffled_ratio);

%% run the algorithm
tic
x_hat = SLR_5_algebraic(A, y);
toc

%% evaluation
error = norm(x_hat - x) / norm(x);

