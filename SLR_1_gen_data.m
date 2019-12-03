function [A, y, x] = SLR_1_gen_data(m, n, sigma, shuffled_ratio)
% y = Pi*A*x + sigma*noise
A=randn(m,n);
B=A
x=randn(n,1);
s=round(m*shuffled_ratio)
noise=sigma*randn(m,1)
rand_index = randperm(m);%将序号随机排列
index=rand_index(1:s)
index1=shuffle(index)
B(index,:)=A(index1,:)
y=B*x+noise

end
