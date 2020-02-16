n_obs = 2450;
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

starting_value = rand([6, 1]);

if constraint_optimization == 0
    options = optimoptions('fminunc', 'Algorithm', 'trust-region',...
        'CheckGradients', true, 'Diagnostics', 'off', ...
        'SpecifyObjectiveGradient', true, 'Display', 'off', ...
        'MaxIterations', 1000, 'OptimalityTolerance', 1e-8, ...
        'HessianFcn', 'objective', 'FunctionTolerance', 1e-8);

    [est,fval,exitflag,output,grad,hessian] = fminunc(@TSPoi_mle, ...
    starting_value, options, delta2, delta3, x, z, y1, y2, y3, gmm);
else
    options = optimoptions('fmincon', 'Algorithm', 'trust-region-reflective',...
        'CheckGradients', false, 'Diagnostics', 'off', ...
        'SpecifyObjectiveGradient', true, 'Display', 'off', ...
        'MaxIterations', 1000, 'OptimalityTolerance', 1e-10, ...
        'HessianFcn', 'objective', 'FunctionTolerance', 1e-10);
    ub = [inf*ones(size(beta1, 1) - 2, 1); 1; 1];
    lb = -ub;
    [est,fval,exitflag,output,lambda,grad,hessian] = fmincon(@TSPoi_mle, ...
        starting_value, [], [], [], [], lb, ub, [], options, ...
        delta2, delta3, x, z, y1, y2, y3, gmm);
end

true_params = [beta1 + [sigma^2/2;zeros(3, 1)]; rho1; rho2];

[V1, V2] = Cal_TSPoi_SE(est, delta2, delta3, ...
    x, z, y1, y2, y3, gmm);

SE1 = sqrt(diag(V1));
SE2 = sqrt(diag(V2));

disp([true_params, est, SE1, SE2])

writematrix([y1, x], 'test_TSPoi_mle.csv')