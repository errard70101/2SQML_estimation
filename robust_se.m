% Robust White Standard Errors for PPML

function F = robust_se(b1, x, zex)

[nexog, kexog]=size(zex);
 
mat1 = inv( zex'*((exp(zex*b1)*ones(1, kexog)).*zex) );
mat2 = zex'*( ( (x-exp(zex*b1)).^2 * ones(1, kexog) ).*zex );
 
F = sqrt(diag(mat1'*mat2*mat1));
end
 
 