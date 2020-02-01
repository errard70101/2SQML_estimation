function [ff, g, H] = biv_mle(starting_value, y2, y3, z2, z3)

% This script contains the loglikelihood, gradient, and hessian of the
% following bivariate probit model:
%     y2 = z2*delta2 + e2,
%     y3 = z3*delta3 + e3,
% where e2 and e3 follow a bivariate normal distribution with mean zeros
% and covariance matrix below
%     [1, gmm; gmm, 1]
%
% y2, y3: dependent variables, n_obs x 1 vectors.
% z2, z3: indenpendent variables, n_obs x n_params vectors.
% starting_value: a vector contains the starting values of parameters to be
%         estimated.
% To note that, the size of z2 and z3 should be identical.
% The formula here bases on Greene's Econometric Analysis 7ed, p739 - p741.

starting_value = starting_value(:);
n_params = size(z2, 2);
n_obs = size(z2, 1);
delta2 = starting_value(1:n_params);
delta3 = starting_value(n_params+1:2*n_params);
gmm = starting_value(end);

y2s = z2*delta2;
y3s = z3*delta3;
q2 = 2*y2 - 1; q3 = 2*y3 - 1;
w2 = q2.*y2s; w3 = q3.*y3s;
clear y2s y3s
rho = q2.*q3.*gmm;
F = zeros(n_obs, 1);
f = zeros(n_obs, 1);
for i = 1:n_obs
    f(i) = mvnpdf([w2(i), w3(i)], zeros(1, 2), [1, rho(i); rho(i), 1]);
    F(i) = mvncdf([w2(i), w3(i)], zeros(1, 2), [1, rho(i); rho(i), 1]);
end
ff = -sum(log(F), 1);


if nargout > 1
    delta = (1./sqrt(1-rho.^2));
    v2 = delta.*(w3 - rho.*w2);
    v3 = delta.*(w2 - rho.*w3);
    
    g2 = normpdf(w2).*normcdf(v2);
    g3 = normpdf(w3).*normcdf(v3);
    d_delta2 = sum((q2.*g2./F).*z2);
    %disp(size(d_delta2))
    d_delta3 = sum((q3.*g3./F).*z3);
    %disp(size(d_delta3))
    d_rho = sum(q2.*q3.*f./F);
    %disp(size(d_rho))

    g = -[d_delta2'; d_delta3'; d_rho];
end
    
if nargout > 2
    wRw = (delta.^2).*(w2.^2 + w3.^2 - 2.*rho.*w2.*w3);
    
    dd_delta2 = -z2'*(z2.*(w2.*g2./F + rho.*f./F + (g2.^2)./(F.^2)));
    %disp(size(dd_delta2))
    dd_delta3 = -z3'*(z3.*(w3.*g3./F + rho.*f./F + (g3.^2)./(F.^2)));
    %disp(size(dd_delta3))
    dd_delta2_delta3 = (q2.*q3.*z2)'*(z3.*(f./F - g2.*g3./(F.^2)));
    %disp(size(dd_delta2_delta3))
    dd_delta2_rho = sum((q3.*z2.*f./F).*(rho.*delta.*v2 - w2 - g2./F));
    %disp(size(dd_delta2_rho))
    dd_delta3_rho = sum((q2.*z3.*f./F).*(rho.*delta.*v3 - w3 - g3./F));
    %disp(size(dd_delta3_rho))
    dd_rho = sum((f./F).*((delta.^2).*rho.*(1 - wRw) + ...
        (delta.^2).*w2.*w3 - f./F));
    %disp(size(dd_rho))

    H = -[dd_delta2, dd_delta2_delta3, dd_delta2_rho'; ...
          dd_delta2_delta3, dd_delta3, dd_delta3_rho'; ...
          dd_delta2_rho, dd_delta3_rho, dd_rho];
end
