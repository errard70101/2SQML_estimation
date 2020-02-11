clear all
load('test_11-Feb-2020_2500.mat')

beta1 = [0.5 + 0.5; 0; 1; -1];
rho1 = 0.3;
rho2 = 0.3;

test1 = sum(abs(b2(:, [1, 2, 3, 4, 103, 104]) - [beta1', rho1, rho2])./SE1(:, [1, 2, 3, 4, 103, 104]) > 1.96);
test2 = sum(abs(b2(:, [1, 2, 3, 4, 103, 104]) - [beta1', rho1, rho2])./SE2(:, [1, 2, 3, 4, 103, 104]) > 1.96);