function [fe] = random_fix_effect(n_obs)
% This function generate random fix effect according to observation numbers
n = sqrt(n_obs);
exporter = zeros(n_obs, 1);
importer = zeros(n_obs, 1);

for i = 1:n
    exporter( (i-1)*n + 1: i*n) = ones(n, 1)*i;
    for j = 1:n
        importer( (i-1)*n + j) = j;
    end
end


group = [exporter importer];

fe = dummyvar(group);