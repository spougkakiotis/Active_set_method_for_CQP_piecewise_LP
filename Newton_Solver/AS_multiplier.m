function [w] = AS_multiplier(x,NS)
% ===================================================================================================================== %
% Normal Equations Operator:
% --------------------------------------------------------------------------------------------------------------------- %
% [w] = AS_multiplier(NS,x) takes as an input the struct containing the Newton blocks, the semismooth Newton projection 
% matrix, as well as  a vector of size m or n, and returns the matrix-vector product of the 
% augmented system matrix by this vector.
% _____________________________________________________________________________________________________________________ %
    n = NS.u_hat_size;
    m = NS.m;
    w = zeros(n+m,1);
    x_1 = x(1:n,1); x_2 = x(n+1:end,1);
    w(1:n,1) = - (NS.Q_bar(:,NS.u_hat_active)*x_1 + (NS.beta+ (1/NS.rho)).*x_1 ...
               - NS.beta.*(NS.B(NS.u_hat_active).*x_1)) + NS.A(:,NS.u_hat_active)'*(x_2);
    w(n+1:end,1) = NS.A(:,NS.u_hat_active)*x_1 + (1/NS.beta).*x_2;
end

