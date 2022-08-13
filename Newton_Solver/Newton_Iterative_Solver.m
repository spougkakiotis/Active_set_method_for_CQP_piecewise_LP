function [dx,dv,iter,instability,drop_direction] = Newton_Iterative_Solver(NS,rhs,PS,maxit,tol,pl)
% ==================================================================================================================== %
% Newton Iterative Solver 
% -------------------------------------------------------------------------------------------------------------------- %
% [dx,iter,instability,drop_direction] = Newton_Iterative_Solver(NS,rhs,PS,maxit,tol)
%
%                                                INPUT: takes as an input all relevant 
%                                                information needed  to build the semismooth Newton system at 
%                                                the j-th iteration of the SSN solver as well as its preconditioner. 
%                                                Then, it employs Preconditioned CG
%                                                to solve the associated linear system.
%
%                                                OUTPUT: it returns the SSN direction dx, the number of inner 
%                                                iterations required to find it, as well as some flag variables 
%                                                indicating ill-conditioning or whether the direction is inaccurate
%                                                and should be dropped.
% 
% Author: Spyridon Pougkakiotis.
% ____________________________________________________________________________________________________________________ %
    accuracy_bound = 1e-1;
    instability = false;
    drop_direction = false;
    tol = min(tol,1e-3);
    dx = zeros(NS.u_hat_size,1);
    dv = zeros(NS.m,1);
    warn_stat = warning;
    warning('off','all');
    [lhs, flag, res, iter] = minres(@(x) AS_multiplier(x,NS), rhs, tol, maxit, @(x) Precond_Operator(x,PS));
    warning(warn_stat);
    if (pl >= 3)
        fprintf(NS.fid,'-------------------------------***Krylov method: MINRES***-----------------------------------------\n');
        fprintf(NS.fid,'Krylov Iterations                     Krylov Flag                      Residual                    \n');
        fprintf(NS.fid,'%4d                                %4d                              %9.2e              \n',iter,flag,res);
        fprintf(NS.fid,'---------------------------------------------------------------------------------------------------\n');
    end

    if (flag > 0) % Something went wrong, so we assume that the preconditioner is not good enough -> increase quality.
        iter = maxit;
        if (res > accuracy_bound)
            drop_direction = true;
            return;
        elseif (flag == 2 || flag == 4 || flag == 5)
            instability = true;
            fprintf('Instability detected during the iterative method. flag = %d.\n',flag);
            return;
        end
    end
    if (nnz(isnan(lhs)) > 0 || nnz(isinf(lhs)) > 0 || (max(lhs) == 0 && min(lhs) == 0)) % Check for ill-conditioning.
        instability = true;
        iter = maxit;
        fprintf('Instability detected during the iterative method.\n');
        return;
    end
    dx = lhs(1:NS.u_hat_size,1);
    dv = lhs(NS.u_hat_size+1:end);
end

