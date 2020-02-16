clear all;
clc;
warning off all;

% This script calculates the counterfactual welfare analysis of in the
% paper "Welfare Gains from WTO and PTA: Count Data Models with Dual
% Endogenous Binary Treatments" version 7.
%% To load data and create summary table
if ispc == 0
    path = '/Volumes/GoogleDrive/§Úªº¶³ºÝµwºÐ/Research Projects/Chen-Lin-Peng-Tsay-2018/';
else
    path = 'G:/§Úªº¶³ºÝµwºÐ/Research Projects/Chen-Lin-Peng-Tsay-2018/';
end
file = 'Processed Data/egger_add_wto.csv';

Dta = readtable(strcat(path, file));

Dta.one_wto = Dta.gatt_d;
Dta.both_wto = Dta.gatt_o .* Dta.gatt_d;
Dta.Y = Dta.exports_x + Dta.intra_x;
Dta.E = Dta.imports_m + Dta.intra_m;

%%
Dta.x_ALB = [];
Dta.m_ALB = [];
%% Means
Dta.X = Dta.X/1000; % Shrink x
Dta.Y = Dta.Y/1000;
Dta.E = Dta.E/1000;
Dta.DIST = Dta.DIST/10;
Dta.DURAB = Dta.DURAB/100;
Dta.POLCOMP = Dta.POLCOMP/10; 
Dta.AUTOC = Dta.AUTOC/10;

var = [Dta.X, Dta.PTA, Dta.DIST, Dta.BORD, ...
       Dta.LANG, Dta.COLONY, Dta.COMCOL, Dta.CURCOL, ...
       Dta.SMCTRY, Dta.CONT, Dta.DURAB, Dta.POLCOMP, ...
       Dta.AUTOC, Dta.gatt_o, Dta.gatt_d, Dta.one_wto, ...
       Dta.both_wto];
means = mean(var)';
sd = std(var)';
minimum = min(var)';
maximum = max(var)';

disp('Summary statistics of variables'); 
disp([means, sd, minimum, maximum]);
% Note: Further variables in data: lnx, durab2, polcomp2, autoc2, dist2, 
% x_x_1, x_x_2,...,x_x_250

%% Defining groups of variables

% Dependent variables: 
%       x ... exports
%       i ... =1 if exports>0
% 
% Groups of regressors:
%       pta ... PTA (regressor in x and in i)
%       z   ... other remaining regressors for x 
%       fe  ... fixed effects: x_x_1 - x_x_250
%       q   ... other remaining regressors for i: z and instruments
%       w   ... regressors for pta: z and instruments

const = ones(size(Dta.X, 1),1);
instr = [Dta.COLONY, Dta.COMCOL, Dta.SMCTRY];
z = [Dta.DIST, Dta.BORD, Dta.LANG, Dta.CONT, Dta.DURAB, Dta.POLCOMP, ...
    Dta.AUTOC, Dta.CURCOL, const];
w = [z(:,1:end-1), instr, const];

VNames = Dta.Properties.VariableNames;
if strcmp(VNames(32), 'x_ARG') == 0 || strcmp(VNames(281), 'm_ZWE') == 0
    disp('Please check the Egger data. The variables are not ordered as expected.')
    return
end
disp(VNames(31 + 59))
disp(VNames(31 + 162))

fe = Dta(:, 32:281);
fe = table2array(fe);

fem = fe;
fem(:, 162) = []; % Deleting _x_162 because of collinearity.
fem(:, 59) = [];  % Idem for _x_59.
wfe = [w fem]; % all the exogenous variables

zfe = [z, fem];

zex = [Dta.PTA, Dta.both_wto, z, fem];

%% Bivariate Probit

% load starting values

load('C:/Users/Shih-Yang Lin/Documents/GitHub/endogenous-pta-and-wto/ppml-2-stage-dual-endogenous/emp_result_1226.mat')
%load('C:/Users/Shih-Yang Lin/Documents/GitHub/endogenous-pta-and-wto/ppml-2-stage-dual-endogenous/estimation_result_0104.mat')

%[gamma_1, ~, ~] = glmfit(wfe, Dta.PTA,'binomial','link','probit','constant','off');
%[gamma_2, ~, ~] = glmfit(wfe, Dta.both_wto,'binomial','link','probit','constant','off');
%gmm = 0;

tic;
f_pre = 10^4;
n_skip = 0;
for iter = 1:1000
starting_value = [gamma_1; gamma_2; gmm];
starting_value = starting_value + rand(length(starting_value), 1) - 0.5;

% bounded optimizer
% algorithm: trust-region-reflective, interior-point
options = optimoptions(@fmincon, 'Algorithm', ...
    'trust-region-reflective', 'FiniteDifferenceType', 'central', ...
    'MaxIterations', 400, 'OptimalityTolerance', 1e-08, ...
    'SpecifyObjectiveGradient', true, 'StepTolerance', 1e-08, ...
    'FunctionTolerance', 1e-8, 'Display', 'iter', ...
    'HessianFcn', 'objective', 'CheckGradients', false);

lb = [-inf(length(starting_value)-1, 1); -ones(1, 1)*1];
ub = [inf(length(starting_value)-1, 1); ones(1, 1)*1];
 
try
[delta, f, exit_FS, ~, ~, ~, H] = ...
   fmincon(@biv_mle, starting_value, [], [], [], [], lb, ub, [], ...
   options, Dta.PTA, Dta.both_wto, wfe, wfe);
catch
    n_skip = n_skip + 1;
end

se = sqrt(diag(inv(H)));
robust_se = cal_robust_biv_se(delta, Dta.PTA, Dta.both_wto, wfe, wfe);

if f < f_pre
    flag = exit_FS;
    f_pre = f;
    delta_pre = delta;
    se_pre = se;
    robust_se_pre = robust_se;
end

disp('gamma 1 is')
disp([delta(1:11), se(1:11), robust_se(1:11)])
disp('gamma 2 is')
disp([delta(length(gamma_1) + 1 : length(gamma_1) + 11), ...
    se(length(gamma_1) + 1 : length(gamma_1) + 11), ...
    robust_se(length(gamma_1) + 1 : length(gamma_1) + 11)])
disp('gmm is')
disp([delta(end), se(end), robust_se(end)])

end

delta = delta_pre;
se = se_pre;
robust_se = robust_se_pre;

toc;

gamma_1 = delta(1:length(gamma_1));
gamma_2 = delta(length(gamma_1) + 1:end - 1);
gmm = delta(end);

save('emp_result_20200216.mat', 'gamma_1', 'gamma_2', 'gmm')

%%  Save estimation result
variable_name = {'DIST'; 'BORD'; 'LANG'; 'CONT'; 'DURAB';...
    'POLCOMP'; 'AUTOC'; 'CURCOL'; 'COLONY'; 'COMCOL'; 'SMCTRY'; 'xi'};

output_table = table(variable_name, [delta(1:11, 1); gmm], ...
    [robust_se(1:11, 1); robust_se(end, 1)], ...
    [delta(length(gamma_1) + 1: length(gamma_1) + 11, 1); gmm], ...
    [robust_se(length(gamma_1) + 1: length(gamma_1) + 11, 1); robust_se(end, 1)], ...
    'VariableNames', {'Var', 'Gamma_1', 'SE_1', 'Gamma_2', 'SE_2'});

writetable(output_table, 'data/estimation_results/end_pta_bprobit_20200213.csv', ...
    'Encoding', 'UTF-8', 'QuoteStrings', true)


%% Possion PML
% load starting values
load('C:/Users/Shih-Yang Lin/Documents/GitHub/endogenous-pta-and-wto/ppml-2-stage-dual-endogenous/estimation_result_0104.mat')

qn = 2; % Use quasi-newton or true-region
if qn == 1
    options = optimoptions('fminunc', 'Algorithm', 'quasi-newton',...
        'CheckGradients', false, 'Diagnostics', 'off', ...
        'SpecifyObjectiveGradient', false, 'Display', 'iter-detailed', ...
        'MaxIterations', 1e7, 'OptimalityTolerance', 1e-8, ...
        'HessianFcn', [], 'FiniteDifferenceType', 'central', ...
        'FiniteDifferenceStepSize', 1e-10, 'MaxFunctionEvaluations', 1e7);
else
    options = optimoptions('fminunc', 'Algorithm', 'trust-region',...
        'CheckGradients', false, 'Diagnostics', 'off', ...
        'SpecifyObjectiveGradient', true, 'Display', 'off', ...
        'MaxIterations', 1000, 'OptimalityTolerance', 1e-8, ...
        'HessianFcn', 'objective', 'FunctionTolerance', 1e-10);
end
exit_TSM = 0;
trial = 1;
f_old = 0;
while trial < 1001
    disp('This is trail')
    disp(trial)
try
 starting_value = b3 + (rand(length(b3), 1) - 0.5) * 2;

 % Need to add Poi_TSM_2end_v2 to this folder
[b3_temp, fval, exit_TSM, ~, grad, Hess] = ...
   fminunc(@Poi_TSM_2end_v2, starting_value, options, gamma_1, gamma_2, ...
   zex, wfe, Dta.X, Dta.PTA, Dta.both_wto, gmm);

if fval < f_old && exit_TSM > 0
    b3 = b3_temp;
    f_old = fval;
end

catch
    disp('failed')
end
trial = trial + 1;
end

save('estimation_result_20190724.mat', 'b3')