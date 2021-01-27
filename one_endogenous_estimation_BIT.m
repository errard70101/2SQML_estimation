clear all;
clc;
%warning off all;

%% To load data and create summary table
if ispc == 0
    path = '/Volumes/GoogleDrive/§Úªº¶³ºÝµwºÐ/Research Projects/Chen-Lin-Peng-Tsay-2018/';
else
    path = 'G:/§Úªº¶³ºÝµwºÐ/Research Projects/Chen-Lin-Peng-Tsay-2018/';
end
file = 'Processed Data/egger_add_bit_20200317.csv';

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
Dta.GDPsum = Dta.GDPsum/100;
Dta.GDPsim = Dta.GDPsim/10;
Dta.REMOTE = Dta.REMOTE/10;
Dta.DROWKL = Dta.DROWKL/10;
Dta.dkl = Dta.dkl/10;

var = [Dta.X, Dta.PTA, Dta.DIST, Dta.BORD, ...
       Dta.LANG, Dta.COLONY, Dta.COMCOL, Dta.CURCOL, ...
       Dta.SMCTRY, Dta.CONT, Dta.DURAB, Dta.POLCOMP, ...
       Dta.AUTOC, Dta.GDPsum, Dta.GDPsim, Dta.REMOTE, ...
       Dta.DROWKL, Dta.dkl, Dta.gatt_o, Dta.gatt_d, ...
       Dta.one_wto, Dta.both_wto, Dta.BIT];
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
%instr = [Dta.COLONY, Dta.COMCOL, Dta.SMCTRY, Dta.GDPsum, Dta.GDPsim, ...
%    Dta.REMOTE, Dta.DROWKL, Dta.dkl];
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

zex = [Dta.BIT, z, fem];
%% Probit of PTA on W 
tic;


%[gamma, gdev, gstatsg] = glmfit(wfe, Dta.BIT,'binomial','link','probit','constant','off');   
%segamma = gstatsg.se;
%V = gstatsg.covb;

[gamma, cov_Hessian, ~, ~, ~] = PROBIT( Dta.BIT , wfe, 3);
segamma = sqrt(diag(cov_Hessian));
V = cov_Hessian;


disp('PROBIT FOR PTA');
disp('coeff         se');
disp([gamma(1:size(z,2)+size(instr,2),1) segamma(1:size(z,2)+size(instr,2),1)]);

wg = wfe*gamma;

toc; t1=toc;

%% Poisson PML of X on PTA, Z
tic;

[b1] = glmfit(zex, Dta.X, 'poisson', 'constant', 'off');   
se1 = robust_se(b1, Dta.X, zex);

disp('PPML WITH EXOGENOUS PTA');
disp('coeff         se1');
disp([b1(1:size(z,2)+1,1) se1(1:size(z,2)+1,1)]);

disp('Average Partial Effect / Partial Effect at Average of PTA:');
ape1=exp(b1(1))-1; 
disp(ape1);

toc; t2=toc;

%% Poisson PML of X on PTA, Z with endogenous PTA
tic;

bin = [b1 ; 0.1];


options=optimset('TolFun',1E-6,'TolX',1e-6,'Jacobian','on'); % 'Display','Iter' (for more fun!)
b2 = fsolve(@foc_psn_new, bin, options, Dta.X, zfe, wfe, Dta.BIT, gamma);
D2 = varmat_new(b2, Dta.X, zfe, wfe, Dta.BIT, gamma, V);
se2 = sqrt(diag(D2));

disp('PPML WITH ENDOGENOUS PTA');
disp('coeff         se');
disp([b2(1:size(z,2)+1,1) se2(1:size(z,2)+1,1)] );
disp([b2(end,1) se2(end,1)] );

disp('Average Partial Effect / Partial Effect at Average of PTA:');
ape2 = exp(b2(1))-1; 
disp(ape2);

toc; t3=toc;