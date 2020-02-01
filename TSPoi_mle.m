function[l, g, H] = TSPoi_mle(starting_value, delta2, delta3, x, z, y1, y2, y3, gmm)
% This script calculates the log-likelihood, gradients, and hessians of the
% second step Poisson estimation.

% y1, y2, and y3 are N x 1 vectors.

% First-step parameters:
% delta2 and delta3 are J x 1 vector.
% gmm is a scalar.

% Second-step parameters:
beta1 = starting_value(1:end-2);
mu2 = starting_value(end - 1);
mu3 = starting_value(end);

% Calculate log-likelihood function
q2 = 2.*y2 - 1; q3 = 2.*y3 - 1;            % N x 1 vector
w2 = q2.*(z*delta2); w3 = q3.*(z*delta3);  % N x 1 vector
gmms = q2.*q3.*gmm;                        % N x 1 vector

SIGMA_2 = [1, gmm; gmm, 1];                % 2 x 2 matrix
SIGMA_2n = [1, -gmm; -gmm, 1];             % 2 x 2 matrix

MU_2 = zeros(1, 2);
X_2 = [w2 + q2.*mu2, w3 + q3.*mu3];
X = [w2, w3];

lambda = exp(x*beta1);
Phi_mu = y2.*y3.*mvncdf(X_2, MU_2, SIGMA_2) ...
    + y2.*(1 - y3).*mvncdf(X_2, MU_2, SIGMA_2n) ...
    + (1 - y2).*y3.*mvncdf(X_2, MU_2, SIGMA_2n) ...
    + (1 - y2).*(1 - y3).*mvncdf(X_2, MU_2, SIGMA_2);
Phi = y2.*y3.*mvncdf(X, MU_2, SIGMA_2) ...
    + y2.*(1 - y3).*mvncdf(X, MU_2, SIGMA_2n) ...
    + (1 - y2).*y3.*mvncdf(X, MU_2, SIGMA_2n) ...
    + (1 - y2).*(1 - y3).*mvncdf(X, MU_2, SIGMA_2);
Psi = Phi_mu./Phi;
lambda_s = lambda.*Psi;

l = -sum((-lambda_s + y1.*log(lambda_s)), 1);

if nargout > 1
    % Calculate derivative of Psi with respect to mu:
    psi_mu2 = q2.*normpdf(w2 + q2.*mu2)./Phi.* ...
        normcdf((w3 + q3.*mu3 - gmms.*(w2 + q2.*mu2))./sqrt(1 - gmms.^2));
    psi_mu3 = q3.*normpdf(w3 + q3.*mu3)./Phi.* ...
        normcdf((w2 + q2.*mu2 - gmms.*(w3 + q3.*mu3))./sqrt(1 - gmms.^2));

    % The gradients
    g_beta1 = sum(x.*(y1 - lambda_s), 1);
    g_mu2 = sum((psi_mu2./Psi).*(y1 - lambda_s), 1);
    g_mu3 = sum((psi_mu3./Psi).*(y1 - lambda_s), 1);
    g = -[g_beta1(:); g_mu2; g_mu3];
end

if nargout > 2
    % Calculate the derivatives of psi_mu with respect to mu:
    psi_mu2_mu2 = -normpdf(w2 + q2.*mu2)./Phi.*(...
        (w2 + q2.*mu2).*normcdf((w3 + q3.*mu3 - gmms.*(w2 + q2.*mu2))./sqrt(1 - gmms.^2)) ...
        + normpdf((w3 + q3.*mu3 - gmms.*(w2 + q2.*mu2))./sqrt(1 - gmms.^2)).* ...
        gmms./sqrt(1 - gmms.^2));
    
    psi_mu3_mu3 = -normpdf(w3 + q3.*mu3)./Phi.*(...
        (w3 + q3.*mu3).*normcdf((w2 + q2.*mu2 - gmms.*(w3 + q3.*mu3))./sqrt(1 - gmms.^2)) ...
        + normpdf((w2 + q2.*mu2 - gmms.*(w3 + q3.*mu3))./sqrt(1 - gmms.^2)).* ...
        gmms./sqrt(1 - gmms.^2));
    
    psi_mu2_mu3 = q2.*q3./Phi.*normpdf(w2 + q2.*mu2)./sqrt(1 - gmms.^2) ...
        .*normpdf((w3 + q3.*mu3 - gmms.*(w2 + q2.*mu2))./sqrt(1 - gmms.^2));
    %psi_mu3_mu2 = q2.*q3./Phi.*normpdf(w3 + q3.*mu3)./sqrt(1 - gmms.^2) ...
    %    .*normpdf((w2 + q2.*mu2 - gmms.*(w3 + q3.*mu3))./sqrt(1 - gmms.^2));
    
    hbb = -x'*(lambda_s.*x);
    hbm2 = -sum(psi_mu2.*lambda.*x, 1);
    hbm3 = -sum(psi_mu3.*lambda.*x, 1);
    hm2m3 = sum(((psi_mu2_mu3.*Psi - psi_mu2.*psi_mu3)./(Psi.^2)).*(y1 - lambda_s)...
        - (psi_mu2.*psi_mu3./Psi).*lambda, 1);
    hm2m2 = sum( ((psi_mu2_mu2.*Psi - psi_mu2.^2)./(Psi.^2)).*(y1 - lambda_s)...
        - ((psi_mu2.^2)./Psi).*lambda, 1);
    hm3m3 = sum( ((psi_mu3_mu3.*Psi - psi_mu3.^2)./(Psi.^2)).*(y1 - lambda_s)...
        - ((psi_mu3.^2)./Psi).*lambda, 1);
    
    H = -[hbb, hbm2', hbm3'; hbm2, hm2m2, hm2m3; ...
        hbm3, hm2m3, hm3m3];
end