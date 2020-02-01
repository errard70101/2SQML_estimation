clear all;
clc;
warning off all;
%% Select hyperparameters
observations = [50^2, 100^2, 150^2];
% number of simulation
m = 100;

%%

for l = 1:size(observations, 2)
    disp(strcat('n_obs =', num2str(observations(l), '%10.0f')))
%%
fix = 1;
egger_fix =0;
n_obs = observations(l); % number of observations
% generate true parameters
true_b = [0.5, 0, 1, -1];
true_d2 = [0, 0.5, -1, 1];
true_d3 = [0, 0.5, 1, -1];
sigma = 1;
rho1 = [0.3, 0.6, 0.9];
rho2 = [0.3, 0.6, 0.9];
gmm = [0.3, 0.6, 0.9];
MU = [0, 0, 0];

if fix == 1 && egger_fix == 0
    fe = random_fix_effect(n_obs(l)); % to construct import and export fix effect dummies
    % exclude the first import and export fix effect dummies to prevent
    % multicollinearity.
    fem = fe;
    fem(:, [1, 1+sqrt(n_obs(l))]) = [];
    beta1 = [true_b, ...
        (rand([1, size(fem, 2)])-0.5).*randn([1, size(fem, 2)])*0.5 ]';
    delta2 = [true_d2, ...
        (rand([1, size(fem, 2)])-0.5).*randn([1, size(fem, 2)])*0.5 ]';
    delta3 = [true_d3, ...
        (rand([1, size(fem, 2)])-0.5).*randn([1, size(fem, 2)])*0.5 ]';
    size_b = length(true_b);
    size_d2 = length(true_d2);
    size_d3 = length(true_d3);
elseif fix == 0 && egger_fix == 0
    fem = [];
    beta1 = true_b(:);
    delta2 = true_d2(:);
    delta3 = true_d3(:);
    size_b = length(true_b);
    size_d2 = length(true_d2);
    size_d3 = length(true_d3);
elseif egger_fix == 1
    head_data = head_data_generator();
    % all we need is the distance variable, head_data.ldis
    x_s = [head_data.ldis, head_data.lYi, head_data.lYn]; % Can include more variables.
    fe = [head_data.dummy_d, head_data.dummy_o];
    fem = fe; fem(:, 162) = []; fem(:, 59) = [];
    x_s = x_s(:, (size(true_b(:), 1) -2));
    pta = head_data.rta;
    clear head_data
    delta2 = [true_d2, (rand([1, size(fem, 2)])-0.5).*randn([1, size(fem, 2)])*0.5 ]';
    delta3 = [true_d3, (rand([1, size(fem, 2)])-0.5).*randn([1, size(fem, 2)])*0.5 ]';
    beta1 = [true_b, (rand([1, size(fem, 2)])-0.5).*randn([1, size(fem, 2)])*0.5 ]';
    size_b = length(true_b);
    size_d2 = length(true_d2);
    size_d3 = length(true_d3);
end

size_bs = size_b + 3;
n_rho = length(rho1);



% I store the estimate result by observations, therefore, these estimates
% holders only need room for m (simulations) x length(parameters)

% store estimates
gamma2 = zeros(m, length(delta2));
gamma3 = zeros(m, length(delta3));
b1 = zeros(m, length(beta1));
b2 = zeros(m, length(beta1)+3);

% store bias of estimates
dta2_error = zeros(m, size_d2*n_rho);
dta3_error = zeros(m, size_d3*n_rho);
bta_error = zeros(m, size_b*n_rho);
bta_s_error = zeros(m, size_bs*n_rho);

% store mean bias
mean_dta2_error = zeros(n_rho, size_d2);
mean_dta3_error = zeros(n_rho, size_d3);
mean_bta_error = zeros(n_rho, size_b);
mean_bta_s_error = zeros(n_rho, size_bs);

% store rmse
rmse_dta2_error = zeros(n_rho, size_d2);
rmse_dta3_error = zeros(n_rho, size_d3);
rmse_bta_error = zeros(n_rho, size_b);
rmse_bta_s_error = zeros(n_rho, size_bs);

% ===================================================================================================
% This is for an output file
fileID = fopen(strcat('test_', date(), '_', num2str(n_obs, '%6.0f'), '.txt'), 'a');


FirstErrorSpec1 = 'Probit 1 & \t %8.4f & \t %8.4f & \t %8.4f & \t %8.4f & & & \\\\ \t \r\n';
FirstRMSESpec = '($1_{st}$ stage) & \t\t (%8.4f) & \t (%8.4f) & \t (%8.4f) & \t (%8.4f) & & & \\\\ \t \r\n';
FirstErrorSpec2 = 'Probit 2 & \t %8.4f & \t %8.4f & \t %8.4f & \t %8.4f & & & \\\\ \t \r\n';

Second_sErrorSpec = 'Poisson & \t %8.4f & \t %8.4f & \t %8.4f & \t %8.4f & \t %8.4f & \t %8.4f & \t %8.4f \\\\ \t \r\n';
Second_sRMSESpec = '($2_{nd}$ stage) & \t\t (%8.4f) & \t (%8.4f) & \t (%8.4f) & \t (%8.4f) & \t (%8.4f) & \t (%8.4f) & \t (%8.4f) \\\\ \t \r\n';
SecondErrorSpec = 'Simple Poisson & \t\t %8.4f & \t %8.4f & \t %8.4f & \t %8.4f & & & \\\\ \t \r\n';
SecondRMSESpec = '& \t\t (%8.4f) & \t (%8.4f) & \t (%8.4f) & \t (%8.4f) & & & \\\\ \t \r\n';

TopRuleSpec = '\\toprule \r\n';
MidRuleSpec = '\\midrule \r\n';
BottomRuleSpec = '\\bottomrule \r\n';


fprintf(fileID, '\\begin{table} \r\n \\centering \r\n');
fprintf(fileID, '\\caption{Simulation Result $\\# observations = %6.0f $} \r\n', n_obs);
fprintf(fileID, '\\scalebox{0.7}{ \r\n \\begin{tabular}{cccccccc} \r\n');
fprintf(fileID, TopRuleSpec);
% ===================================================================================================
tic
for r = 1:n_rho
    disp(strcat('rho1 =', num2str(rho1(r), '%2.2f')))
    disp(strcat('rho2 =', num2str(rho2(r), '%2.2f')))
    disp(strcat('gamma =', num2str(gmm(r), '%2.2f')))
    SIGMA = [sigma^2, sigma*rho1(r), sigma*rho2(r);...
        sigma*rho1(r), 1, gmm(r); ...
        sigma*rho2(r), gmm(r), 1];
    tried = 0;
    textprogressbar('Calculating outputs: ');
    for i = 1:m
        [gamma2_temp, gamma3_temp, b1_temp, b2_temp, tried] = ...
            mon_two_end_pross(n_obs, beta1, delta2, delta3, MU, SIGMA, ...
            fem, fix, tried); % default egger_fix = 0
        gamma2(i, :) = gamma2_temp';
        gamma3(i, :) = gamma3_temp';
        b1(i, :) = b1_temp;
        b2(i, :) = b2_temp;
        textprogressbar(i*100/m);
    end
    textprogressbar('Done!');
    clear textprogressbar i
    disp(strcat('Sucessful rate:', num2str(m*100/tried), '%'))

    % calculate the estimation errors and store them (not store estimates
    % of fix effects.)
    temp = gamma2 - delta2';
    dta2_error(:, (r-1)*size_d2 + 1:r*size_d2) = temp(:, 1:size_d2);
    clear temp
    temp = gamma3 - delta3';
    dta3_error(:, (r-1)*size_d3 + 1:r*size_d3) = temp(:, 1:size_d2);
    clear temp
    true_2 = [beta1; sigma*rho1(r); sigma*rho2(r); gmm(r)];
    true_2(1) = true_2(1) + sigma^2/2;
    temp =  b1 - true_2(1:end-3)';
    bta_error(:, (r-1)*size_b + 1:r*size_b) = temp(:, 1:size_b);
    clear temp
    temp = b2 - true_2';
    bta_s_error(:, (r-1)*size_bs + 1:r*size_bs) = [temp(:, 1:size_b), temp(:, end-2:end)];
    clear temp
    
    mean_dta2_error(r, :) = mean(dta2_error(:, (r-1)*size_d2 + 1:r*size_d2), 1);
    mean_dta3_error(r, :) = mean(dta3_error(:, (r-1)*size_d3 + 1:r*size_d3), 1);
    mean_bta_error(r, :) = mean(bta_error(:, (r-1)*size_b + 1:r*size_b), 1);
    mean_bta_s_error(r, :) = ...
        mean(bta_s_error(:, (r-1)*size_bs+1: r*size_bs), 1);

    rmse_dta2_error(r, :) = sqrt(mean(dta2_error(:, (r-1)*size_d2 + 1:r*size_d2).^2, 1));
    rmse_dta3_error(r, :) = sqrt(mean(dta3_error(:, (r-1)*size_d3 + 1:r*size_d3).^2, 1));
    rmse_bta_error(r, :) = sqrt(mean(bta_error(:, (r-1)*size_b + 1:r*size_b).^2, 1));
    rmse_bta_s_error(r, :) = ...
        sqrt(mean(bta_s_error(:, (r-1)*size_bs + 1:r*size_bs).^2, 1));
    
    
    fprintf(fileID, 'Model %1.0f & \\multicolumn{7}{c}{$\\rho_1 = %2.1f$, $\\rho_2 = %2.1f$, $\\gamma = %2.1f$} \\\\ \r\n', r, rho1(r), rho2(r), gmm(r));
    fprintf(fileID, '& $Constant$ & $\\delta_{j1}$ & $\\delta_{j2}$ & $\\delta_{3j}$ & & & \\\\ \r\n');
    fprintf(fileID, MidRuleSpec);
    fprintf(fileID, FirstErrorSpec1, mean_dta2_error(r, 1:size_d2));
    fprintf(fileID, FirstRMSESpec, rmse_dta2_error(r, 1:size_d2));
    fprintf(fileID, FirstErrorSpec2, mean_dta3_error(r, 1:size_d3));
    fprintf(fileID, FirstRMSESpec, rmse_dta3_error(r, 1:size_d3));
    fprintf(fileID, '[1em] \r\n');
    fprintf(fileID, '& $Constant$ & $\\beta_{1}$ & $\\beta_{2}$ & $\\beta_{3}$ & $\\sigma\\rho_1$ & $\\sigma\\rho_2$ & $\\gamma$ \\\\ \r\n');
    fprintf(fileID, MidRuleSpec);
    fprintf(fileID, Second_sErrorSpec, [mean_bta_s_error(r, 1:size_b), mean_bta_s_error(r, end-2:end)]);
    fprintf(fileID, Second_sRMSESpec, [rmse_bta_s_error(r, 1:size_b), rmse_bta_s_error(r, end-2:end)]);
    fprintf(fileID, SecondErrorSpec, mean_bta_error(r, 1:size_b));
    fprintf(fileID, SecondRMSESpec, rmse_bta_error(r, 1:size_b));
    fprintf(fileID, MidRuleSpec);
    
end
toc
%%

fprintf(fileID, BottomRuleSpec);
fprintf(fileID,'\\multicolumn{8}{l}{\\footnotesize Top value in each cell is the mean error (based on 1,000 repetitions). RMSE in parentheses.}\\\\ \r\n');
fprintf(fileID,'\\end{tabular}} \r\n \\end{table} \r\n');
fprintf(fileID,strcat('Elapsed time is ', num2str(toc), ' seconds.'));
fclose(fileID);

%% Save Monte Carlo Data
filename = strcat('test_', date(), '_', num2str(n_obs), '.mat');
save(filename, 'gamma2', 'gamma3', 'b1', 'b2')

end