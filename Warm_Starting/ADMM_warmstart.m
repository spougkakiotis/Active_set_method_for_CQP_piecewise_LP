function [x,w,v,z,opt,iter] = ADMM_warmstart(Q,D,A,A_tr,C,C_tr,b,d,c_1,c_2,lb,ub,tol,maxit)
% ======================================================================================================================== %
% ADMM Warm-start:
% ------------------------------------------------------------------------------------------------------------------------ %
% [x,y,y_2,z,opt,iter] = ADMM_warmstart(Q,D,A,A_tr,C,C_tr,b,d,c_1,c_2,lb,ub,tol,maxit)
%                                           takes as an input the problem data, and applies
%                                           proximal ADMM to find an approximate solution
%                                           of the following primal-dual problem:
%
%     min_{x,w,u}     c^T x +(1/2)x^T Q x + sum_{j=n+1}^{n+l} max(u(j),0) + ||D(u(1:n))||_1 + delta_{K}(u(1:n)),       
%     s.t.            Ax = b, 
%                     Cx + d - w =0,
%                     u = [x; w],
%
%  where g(x) = \|Dx\|_1,  with tolerance tol. It terminates after maxit iterations 
%  and returns an approximate primal-dual solution (x,w,y,z).
%                                                           
% Author: Spyridon Pougkakiotis, New Haven, CT, USA, 2022.
% ________________________________________________________________________________________________________________________ %
    % ==================================================================================================================== %
    % Initialize parameters and relevant statistics.
    % -------------------------------------------------------------------------------------------------------------------- %
    n = size(c_1,1);
    m = size(b,1);
    l = size(c_2,1);
    sigma = 1;                                                                   % Penalty parameter of the proximal ADMM.
    sigma_hat = 0.1;
    gamma = 1;                                                                   % ADMM step-length.
    x = zeros(n,1);  w = zeros(l,1);  u = zeros(l+n,1);  y = zeros(m+n+2*l,1);   % Starting point for ADMM.  
    iter = 0;   opt = 0;
    % ____________________________________________________________________________________________________________________ %
    
    % ==================================================================================================================== %
    % Compute residuals (for termination criteria) and the pADMM penalty parameter.
    % -------------------------------------------------------------------------------------------------------------------- %
    temp_compl_x = u(1:n) + y(l+m+1:l+m+n);
    temp_compl_x = max(abs(temp_compl_x)-D, zeros(n,1)).*sign(temp_compl_x);  % Proximity operator of |Dx|.
    temp_lb = (temp_compl_x < lb);
    temp_ub = (temp_compl_x > ub);
    temp_compl_x(temp_lb) = lb(temp_lb);                                      % Euclidean projection to K.
    temp_compl_x(temp_ub) = ub(temp_ub);
    temp_compl_w = u(n+1:n+l) + y(l+m+n+1:2*l+m+n);                           % Proximity operator of max(x,0).
    temp_compl_w = max(temp_compl_w-ones(l,1),zeros(l,1)) ...
                   + min(temp_compl_w,zeros(l,1));
    compl = norm([u(1:n)-temp_compl_x;u(n+1:n+l)-temp_compl_w]);              % Measure of the complementarity.
    res_p = [(b-A*x); (d+C*x-w); (u - [x;w])];                                % Primal residual
    res_d = [(c_1+Q*x-[C_tr A_tr -speye(n)]*y(1:l+m+n));  ...                 % Dual residual.
             (c_2 + y(1:l) + y(l+m+n+1:2*l+m+n))];
         
    W = [C -speye(l); A sparse(m,l); -speye(l+n)];
    F_M = [[(-gamma).*(Q+sigma_hat.*speye(n)) sparse(n,l); sparse(l,n) -sigma_hat.*speye(l)]         W'; ...
                             W                                                 (1/(gamma*sigma)).*speye(n+m+2*l)];
    [L,D_L,pp] = ldl(F_M,'lower','vector');
    % ____________________________________________________________________________________________________________________ %

    while(iter < maxit)
        % ================================================================================================================ %
        % Check termination criteria.
        % ---------------------------------------------------------------------------------------------------------------- %
        if (norm(res_p)/(1+norm(b)+norm(d)) < tol && norm(res_d)/(1+norm(c_1)+norm(c_2)) < tol ...
             &&  compl/(1 + norm(u) + norm(y(l+m+1:2*l+m+n))) < tol )
            fprintf('optimal solution found\n');
            opt = 1;
            break;
        end
        iter = iter+1;
        % ________________________________________________________________________________________________________________ %
        
        % ================================================================================================================ %
        % 1st sub-problem: calculation of u_{j+1} (prox evaluation of g() and then projection to K \times R^l). 
        % ---------------------------------------------------------------------------------------------------------------- %
        % First n components with g_1 and projection.
        u(1:n) = x + (1/sigma).*y(l+m+1:l+m+n);
        u(1:n) = max(abs(u(1:n))-(1/sigma).*D, zeros(n,1)).*sign(u(1:n));
        u_lb = (u(1:n) < lb);
        u_ub = (u(1:n) > ub);
        u(u_lb) = lb(u_lb);
        u(u_ub) = ub(u_ub);
        % Remaining l components with prox of g_2.
        u(n+1:n+l) = w + (1/sigma).*y(l+m+n+1:2*l+m+n);
        u(n+1:n+l) = max(u(n+1:n+l) - 1/sigma,0) + min(u(n+1:n+l),0);
        % ________________________________________________________________________________________________________________ %
        
        % ================================================================================================================ %
        % 2nd sub-problem: calculation of x_{j+1}, w_{j+1}.
        % ---------------------------------------------------------------------------------------------------------------- %
        rhs = [gamma.*([c_1;c_2] - sigma_hat.*[x;w]) + (1-gamma).*(W'*y); ...
               (1/(sigma*gamma)).*y + [-d;b;-u]];
        warn_stat = warning;
        warning('off','all');
        lhs = L'\(D_L\(L\(rhs(pp))));
        warning(warn_stat);
        lhs(pp) = lhs;
        x = lhs(1:n);   w = lhs(n+1:n+l);   y = lhs(n+l+1:end); 
        % ________________________________________________________________________________________________________________ %
                
        % ================================================================================================================ %
        % Residual Calculation.
        % ---------------------------------------------------------------------------------------------------------------- %       
        temp_compl_x = u(1:n) + y(l+m+1:l+m+n);
        temp_compl_x = max(abs(temp_compl_x)-D, zeros(n,1)).*sign(temp_compl_x);  % Proximity operator of |Dx|.
        temp_lb = (temp_compl_x < lb);
        temp_ub = (temp_compl_x > ub);
        temp_compl_x(temp_lb) = lb(temp_lb);                                      % Euclidean projection to K.
        temp_compl_x(temp_ub) = ub(temp_ub);
        temp_compl_w = u(n+1:n+l) + y(l+m+n+1:2*l+m+n);                           % Proximity operator of max(x,0).
        temp_compl_w = max(temp_compl_w-ones(l,1),zeros(l,1)) ...
                        + min(temp_compl_w,zeros(l,1));
        compl = norm([u(1:n)-temp_compl_x;u(n+1:n+l)-temp_compl_w]);              % Measure of the complementarity.
        res_p = [(b-A*x); (d+C*x-w); (u-[x;w])];                                  % Primal residual
        res_d = [(c_1+Q*x-[C_tr A_tr -speye(n)]*y(1:l+m+n));  ...                 % Dual residual.
                 (c_2 + y(1:l) + y(l+m+n+1:2*l+m+n))];
        % ________________________________________________________________________________________________________________ %
    end
    fprintf('ADMM iterations: %5d\n', iter);
    fprintf('primal  feasibility: %8.2e\n', norm(res_p));
    fprintf('dual feasibility: %8.2e\n', norm(res_d));
    fprintf('complementarity: %8.2e\n', compl); 
    z = retrieve_reformulated_z(D,u(1:n),y(m+l+1:m+l+n));                         % dual slack variables
    v = y(1:m+l);                                                                 % Lagrange multipliers
end


