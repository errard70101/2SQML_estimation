% This script checks if the analytic gradients and hessians are correct by
% numerical method.

% Generate simulated data.

n_obs = 1;
constraint_optimization = 1;

% generate true parameters
beta1 = [0.5; 0; 1; -1];
delta2 = [0; 0.5; -1; 1];
delta3 = [0; 0.5; 1; -1];
sigma = 1;
rho1 = 0.3;
rho2 = 0.3;
gmm = 0.3;
MU = [0, 0, 0];
SIGMA = [sigma^2, sigma*rho1, sigma*rho2;...
        sigma*rho1, 1, gmm; ...
        sigma*rho2, gmm, 1];

% Generate simulation data
R = mvnrnd(MU, SIGMA, n_obs);

x_s = rand([n_obs, (size(beta1, 1) - 3)]) - 0.5;
z_s = rand([n_obs, (size(delta2, 1) - size(x_s, 2) - 1)]) - 0.5;
z = [ones(n_obs, 1), x_s, z_s];
y2 = (z*delta2 + R(:, 2) > 0);
y3 = (z*delta3 + R(:, 3) > 0);
x = [ones(n_obs, 1), x_s, y2, y3];

y1 = exp(x*beta1 + R(:, 1));