function F = varmat_new(b4,exports,xoir,z,rta, gamma, V)

alpha4 = b4(1);
beta4  = b4(2:end-1);
theta4 = b4(end);
psi =    (rta.*normcdf(theta4+z*gamma)./(normcdf(z*gamma))+...
        (1-rta).*(1-normcdf(theta4+z*gamma))./(1-normcdf(z*gamma)));
lw =    exp(xoir*beta4+alpha4.*rta);
lambda = lw.*psi;
dpsi=   normpdf(theta4+z*gamma) .* ( rta./normcdf(z*gamma) - (1-rta)./(1-normcdf(z*gamma)) );  % dpsi = partial derivative of psi w.r.t theta

% Constructing two-step-adjusted heteroskedasticity robust QML s.e.
%V   = gstatsg.covb;     % variance of gamma
res   = (exports - lambda); % residuals
PSI = diag(res.^2);
G1  = [xoir, rta, dpsi./psi ];  
B = G1'*PSI*G1;
clear PSI;
LAMBDA = diag(lambda);
A = -G1'*LAMBDA*G1;
clear LAMBDA;
Ainv = inv(A);

test_a = ( normpdf(theta4+z*gamma)./normcdf(z*gamma)-normcdf(theta4+z*gamma)./normcdf(z*gamma).^2.*normpdf(z*gamma) );
test_b = ( -normpdf(theta4+z*gamma)./(1-normcdf(z*gamma))+(1-normcdf(theta4+z*gamma))./...
        (1-normcdf(z*gamma)).^2.*normpdf(z*gamma) );

test_a(rta == 0) = 0;
test_b( (1 - rta) == 0) = 0;
    
psig =  rta .* test_a...
        + (1-rta) .* test_b; % psig = partial derivative of psi w.r.t gamma, but without z (has to be multiplied by z, we do it in the next line)
G2 = kron(ones(1,size(z,2)),lw) .* z.* kron(ones(1,size(z,2)),psig);
C = G1'*G2*V*G2'*G1;
D = Ainv*(B+C)*Ainv;
BB = D(1:end-2,1:end-2); 
AA = D(end-1,end-1);
TT = D(end,end);
AT = D(end,end-1);
BA = D(end-1,1:end-2);
BT = D(end,1:end-2);

F = [AA BA AT; BA' BB BT'; AT BT TT];
end