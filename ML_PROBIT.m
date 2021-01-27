%-----------------------------------
%  ML_PROBIT
%  Usage:
%  Y = dependent variable (n * 1 vector)
%  X = regressors (n * k matrix)
%  method_flag = numeric gradients (1), 
%  analytic gradients (2, default), analytic Hessian (3)
%-----------------------------------
function [log_like,Gradient_c,Hessian_c] = ML_PROBIT(c, Y, X, method_flag)

q = 2*Y-1;
probit_F = normcdf(q.*(X*c));
log_like = sum(log(probit_F));
log_like = -log_like;


if method_flag >= 2    
    lambda = q.*normpdf(q.*(X*c))./probit_F;
    Gradient_c = lambda'*X;
    Gradient_c = -Gradient_c;
end

if method_flag == 3        
    [nobs,nreg] = size(X); 
    Hessian_c = transpose(lambda.*(lambda+X*c)*ones(1,nreg).*X)*X;     
    Hessian_c = -Hessian_c;
end

end