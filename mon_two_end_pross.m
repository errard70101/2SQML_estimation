function [gamma2, gamma3, b1, b2, tried, V1, V2] = mon_two_end_pross(n_obs, beta1, ...
    delta2, delta3, MU, SIGMA, fem, fix, tried)
% This function is to generate random observations and estimate two probit
% models, and use the results as first step estimation to conduct a second
% step poisson estimation. A simple poisson model is conducted as
% comparable group.

% Fist step models:
% y2 = 1[z*delta2 > 0]
% y3 = 1[z*delta3 > 0]

% Second step model:
% y1 = exp(x*beta1)

exit_TSM = -1;

while exit_TSM <= 0
    tried = tried + 1;
    
try
%%
egger_fix = 0;
beta1 = beta1(:);
delta2 = delta2(:);
delta3 = delta3(:);
R = mvnrnd(MU, SIGMA, n_obs);

if fix == 1 && egger_fix == 0
    x_s = rand([n_obs, (size(beta1, 1) - size(fem, 2) - 3)]) - 0.5;
    z_s = rand([n_obs, size(delta2, 1) - size(x_s, 2) - size(fem, 2) - 1]) - 0.5;
    z = [ones(n_obs, 1), x_s, z_s, fem];
    y2 = (z*delta2 + R(:, 2) > 0);
    y3 = (z*delta3 + R(:, 3) > 0);
    x = [ones(n_obs, 1), x_s, y2, y3, fem];
elseif fix == 0 && egger_fix == 0
    x_s = rand([n_obs, (size(beta1, 1) - 3)]) - 0.5;
    z_s = rand([n_obs, (size(delta2, 1) - size(x_s, 2) - size(fem, 2) - 1)]) - 0.5;
    z = [ones(n_obs, 1), x_s, z_s];
    y2 = (z*delta2 + R(:, 2) > 0);
    y3 = (z*delta3 + R(:, 3) > 0);
    x = [ones(n_obs, 1), x_s, y2, y3];
elseif egger_fix == 1
    head_data = head_data_generator();
    % all we need is the distance variable, head_data.ldis
    x_s = [head_data.ldis, head_data.lYi, head_data.lYn]; % Can include more variables.
    x_s = x_s(:, length(beta1) -3);
    clear head_data
    z_s = rand([n_obs, length(delta2) - size(x_s, 2) - size(fem, 2) - 1]) - 0.5;
    z = [ones(n_obs, 1), x_s, z_s, fem];
    y2 = (z*delta2 + R(:, 2) > 0);
    y3 = (z*delta3 + R(:, 3) > 0);
    x = [ones(n_obs, 1), x_s, y2, y3, fem];
end

y1 = exp(x*beta1 + R(:, 1));

%% Probit_1 of PTA on W 

[gamma2, ~, gstatsg] = glmfit(z, y2,'binomial','link','probit','constant','off');   
%V = gstatsg.covb;

disp_probit = 0; % Display estimation results?
if disp_probit == 1
    segamma = gstatsg.se;
    disp('PROBIT 1');
    disp('coeff         se');
    disp([gamma2(1:4), segamma(1:4)]);
end
clear gstatsg
%% Probit_2 of PTA on W 

[gamma3, ~, gstatsg] = glmfit(z, y3,'binomial','link','probit','constant','off');   
%V = gstatsg.covb;

disp_probit = 0; % Display estimation results?
if disp_probit == 1
    segamma = gstatsg.se;
    disp('PROBIT 2');
    disp('coeff         se');
    disp([gamma3(1:4), segamma(1:4)]);
end
clear gstatsg
%% Bivariate Probit

% Uncomment the code below to use random starting value
 starting_value = [gamma2(:); gamma3(:); rand(1)];
% bounded optimizer
% algorithm: trust-region-reflective, interior-point
options = optimoptions(@fmincon, 'Algorithm', 'trust-region-reflective', ... 
   'FiniteDifferenceType', 'central', 'MaxIterations', 1e10, ...
   'OptimalityTolerance', 1e-10, 'SpecifyObjectiveGradient', true, ...
   'StepTolerance', 1e-10, 'FunctionTolerance', 1e-10, ...
   'Display', 'off', 'HessianFcn', 'objective', 'CheckGradients', false, ...
   'MaxFunctionEvaluations', 1e10, 'UseParallel', true);
 lb = [-inf(length(starting_value)-1, 1); -1];
 ub = [inf(length(starting_value)-1, 1); 1];
[delta, ~, exitflag_biv, ~, g_temp, H_temp] = ...
   fmincon(@biv_mle, starting_value, [], [], [], [], lb, ub, [], ...
   options, y2, y3, z, z);

delta2 = delta(1: length(gamma2));
delta3 = delta(length(gamma2) + 1: length(gamma2) + length(gamma3));
gmm = delta(end);




%% Simple Poisson

b1 = glmfit(x, y1, 'poisson', 'constant', 'off');
se1 = robust_se(b1, y1, x); %need to rewrite this

disp_poisson = 0; % Display estimation results?
if disp_poisson == 1
    disp('PPML');
    disp('coeff         se');
    disp([b1(1:4), se1(1:4)] );
end

%% Poisson PML of X on PTA, Z with endogenous PTA
constraint_optimization = 1;
starting_value = [b1; ones(2, 1)*0.1] + (rand(size([b1; zeros(2, 1)])) - 0.5)*0.1;

if constraint_optimization == 0
    options = optimoptions('fminunc', 'Algorithm', 'trust-region',...
        'CheckGradients', false, 'Diagnostics', 'off', ...
        'SpecifyObjectiveGradient', true, 'Display', 'off', ...
        'MaxIterations', 1000, 'OptimalityTolerance', 1e-8, ...
        'HessianFcn', 'objective', 'FunctionTolerance', 1e-8);

    [b2, fval, exitflag, output, grad, hessian] = fminunc(@TSPoi_mle, ...
    starting_value, options, delta2, delta3, x, z, y1, y2, y3, gmm);
else
    options = optimoptions('fmincon', 'Algorithm', 'trust-region-reflective',...
        'CheckGradients', false, 'Diagnostics', 'off', ...
        'SpecifyObjectiveGradient', true, 'Display', 'off', ...
        'MaxIterations', 1000, 'OptimalityTolerance', 1e-10, ...
        'HessianFcn', 'objective', 'FunctionTolerance', 1e-10, ...
        'FiniteDifferenceType', 'central', 'UseParallel', true);
    ub = [inf*ones(size(beta1, 1) - 2, 1); 1; 1];
    lb = -ub;
    [b2, fval, exitflag_poi, output, lambda, grad, hessian] = fmincon(@TSPoi_mle, ...
        starting_value, [], [], [], [], lb, ub, [], options, ...
        delta2, delta3, x, z, y1, y2, y3, gmm);
end    

exit_TSM = exitflag_biv * exitflag_poi;

[V1, V2] = Cal_TSPoi_SE(b2, delta2, delta3, x, z, y1, y2, y3, gmm);

b2 = [b2; gmm];

catch
    %disp('Error, estimate again.')
end

end