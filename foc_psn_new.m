function [F, J] = foc_psn_new(bin,y,x,z,rta,gamma)

% gamma is the estimate of probit
% rta is pta
% bin use the exogenous Poisson PML estimate as initial value
% y is export
% x is zfe = [z, fem] % exclude iv
% z is wfe = [w, fem] % include iv


% First Order Conditions of PPML with endog. RTA (score)

alpha=bin(1); % estimate of dist
beta=bin(2:end-1); % estimate of the rest
theta=bin(end); % estimate of constant

lw      = exp(x*beta+alpha.*rta); % expected export
psi     = (rta.*normcdf(theta+z*gamma)./(normcdf(z*gamma))+...
           (1-rta).*(1-normcdf(theta+z*gamma))./(1-normcdf(z*gamma)));
dpsi    = (rta.*normpdf(theta+z*gamma)./(normcdf(z*gamma))+(1-rta).*(-normpdf(theta+z*gamma))./(1-normcdf(z*gamma)));
lambda  = lw.*psi;

F = [ rta'*(y-lambda); x'*(y-lambda); (dpsi./psi)'*(y-lambda) ];
% Evaluate the Jacobian if nargout > 1
if nargout > 1
        Hbb = -x'*((lambda*ones(1,size(x,2))).*x);
        Hba = -x'*(lambda.*rta);
        Hbt = -x'*(lw.*dpsi);
        Haa = -lambda'*rta; 
        Hat = sum(-rta.*(lw.*dpsi));
        % Htt = sum(-lambda.*(dpsi./psi).^2); % if expected hessian
        Htt = sum( -lw.*(dpsi.^2)./psi - (y-lambda).*((theta + z*gamma).*dpsi./psi + (dpsi./psi).^2 ) ); % if actual hessian
        J = [Haa Hba' Hat; Hba Hbb Hbt; Hat Hbt' Htt];  
end
  
% options=optimset('TolFun',1E-6,'TolX',1e-6,'MaxFunEvals',1000000,'MaxIter',10000,...
%                 'Display','Iter','Jacobian','on');
% [xb3,fval3,exitflag3] = fsolve(@foc_psn, binit, options, exports,x,z,rta,gamma);