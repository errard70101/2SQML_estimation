clear all
load('test_13-Feb-2020_2500_0.6.mat')

beta1 = [0.5 + 0.5; 0; 1; -1];
rho1 = 0.6;
rho2 = rho1;

disp('The mean estimates are')
disp(mean(b2(:, [1, 2, 3, 4, end-2, end-1]) - [beta1', rho1, rho2]))
disp(mean(SE1(:, [1, 2, 3, 4, end-1, end])))
disp(mean(SE2(:, [1, 2, 3, 4, end-1, end])))
disp('The standard error calculated from the simulation')
disp(std(b2(:, [1, 2, 3, 4, end-2, end-1])))


test1 = sum(abs(b2(:, [1, 2, 3, 4, end-2, end-1]) - [beta1', rho1, rho2])...
    ./SE1(:, [1, 2, 3, 4, end-1, end]) > 1.96)./size(b1, 1).*100;
test2 = sum(abs(b2(:, [1, 2, 3, 4, end-2, end-1]) - [beta1', rho1, rho2])...
    ./SE2(:, [1, 2, 3, 4, end-1, end]) > 1.96)./size(b1, 1).*100;

test3 = sum(abs(b2(:, [1, 2, 3, 4, end-2, end-1]) - mean(b2(:, [1, 2, 3, 4, end-2, end-1])))...
    ./std(b2(:, [1, 2, 3, 4, end-2, end-1])) > 1.96)./size(b1, 1).*100;

k = 6;
figure()
histogram(b2(:, k), 'Normalization','pdf')
hold on
mu = mean(b2(:, k));
sigma = std(b2(:, k));
y = mu - 4*sigma:0.01:mu + 4*sigma;
f = exp(-(y-mu).^2./(2*sigma^2))./(sigma*sqrt(2*pi));
plot(y,f,'LineWidth',1.5)

figure()
histogram(SE1(:, k), 'Normalization','pdf')
hold on
mu = mean(SE1(:, k));
sigma = std(SE1(:, k));
y = mu - 4*sigma:0.001:mu + 4*sigma;
f = exp(-(y-mu).^2./(2*sigma^2))./(sigma*sqrt(2*pi));
plot(y,f,'LineWidth',1.5)