function [x,w,v,iter,tol_achieved,num_of_factor] = Semismooth_Newton(NS,tol,maxit,pl)
% =================================================================================================================================== %
% Semismooth_Newton: 
% ----------------------------------------------------------------------------------------------------------------------------------- %
% x = Semismooth_Newton(NS,tol,maxit,method,version,LRS)
%                                     takes as an input a MATLAB struct NS that containing relevant information needed 
%                                     to build the semismooth Newton system corresponding to minimizing the augmented 
%                                     Lagrangian, the minimum accuracy tolerance and the maximum number of SSN iterations, 
%                                     the version of SSN-PMM used, as well as a structure for potential low-rank updates. It 
%                                     employs a semismooth Newton iteration (given the current iterate (x_k,y_k,z_k))   
%                                     and returns the accepted "optimal" solution x, v, y.
%
% 
% Author: Spyridon Pougkakiotis, New Haven, CT, USA, 2022.
% ___________________________________________________________________________________________________________________________________ %
    n = NS.n;
    m = NS.m;
    l = NS.l;
    if (nargin < 4 || isempty(pl))    
        pl = 1;    
    end
    x = NS.x;  w = NS.w;  v = NS.v; % Starting point for SSN -> x_0 = x_k, w_0 = w_k, v_0 = v_k.
    % =============================================================================================================================== %
    % Set the semismooth Newton parameters
    % ------------------------------------------------------------------------------------------------------------------------------- %
    eta_1 = (1e-1)*tol;             % Maximum tolerance allowed when solving the corresponding linear systems.
    eta_2 = 0.1;                    % Determines the rate of convergence of SSN (that is, the rate is (1+eta_2)).
                                    % The trade-off: for larger eta_2, SSN is faster but CG is slower. (eta_2 in (0,1].)
    mu = (0.4995/2);                % Fraction of the decrease in Lagrangian predicted by linear extrapolation that we accept.
    delta = 0.995;                  % Maximum possible step-length used within the backtracking line-search.
    % _______________________________________________________________________________________________________________________________ %
    
    % =============================================================================================================================== %
    % Initialize metrics.
    % ------------------------------------------------------------------------------------------------------------------------------- %
    iter = 0;                       % keep track of SSN iterations
    num_of_factor = 0;              % keep track of the number of factorizations performed
    max_outliers = 0;               % maximum number of outliers allowed before recomputing the factorization
    dx = zeros(n,1);
    dw = zeros(l,1);
    dv = zeros(l+m,1);
    % _______________________________________________________________________________________________________________________________ %
 
    while (iter < maxit)
    % ------------------------------------------------------------------------------------------------------------------------------- %
    % SSN Main Loop structure:
    % Until (|| M(x_{k+1},w_{k+1},y_{k+1},z_k) || <= tol) do
    %   Build matrices B_delta, B_g_1, B_g_2, belonging in the Clarke subdifferential of Proj_{K}(beta^{-1} z_k +  x_j),
    %   and in the Clarke subdifferential of prox_{zeta g}([x_j;w_j] - zeta res_smooth_primal), respectively.
    %   Let matrix M_j in the Clarke subdifferential of M(,,,) and compute an spd preconditioner if necessary
    %   Approximately solve the system: M_j d = - M(x_{k_j},w_{k_j},y_{k_j},z_k), using minres or factorization.
    %   Perform Line-search with Backtracking
    %   j = j + 1;
    % End
    % ------------------------------------------------------------------------------------------------------------------------------- %
        % =========================================================================================================================== %
        % Compute and store an element in the Clarke subdifferential of Proj_{K}(beta^{-1} z_k + x_j). 
        % --------------------------------------------------------------------------------------------------------------------------- %
        proj_vec = (1/NS.beta).*(NS.z) + x;  
        tmp_lb = (proj_vec < NS.lb);
        tmp_ub = (proj_vec > NS.ub);
        proj_vec(tmp_lb) = NS.lb(tmp_lb);
        proj_vec(tmp_ub) = NS.ub(tmp_ub);
        B_delta = ones(n,1);
        B_delta(tmp_lb) = zeros(nnz(tmp_lb),1);
        B_delta(tmp_ub) = zeros(nnz(tmp_ub),1);
        NS.B_delta = B_delta;
        % ___________________________________________________________________________________________________________________________ %

        % =========================================================================================================================== %
        % Compute and store an element in the Clarke subdifferential of prox_{zeta g}(x_j - zeta res_smooth_primal). 
        % --------------------------------------------------------------------------------------------------------------------------- %
        res_smooth_x = NS.c_1 + NS.Q*x - [NS.C_tr NS.A_tr]*v+ NS.z + NS.beta.*x - NS.beta.*proj_vec + (1/NS.rho).*(x-NS.x);
        res_smooth_w = NS.c_2 + v(1:l);
        u_g_1 = x - NS.zeta.*res_smooth_x;
        u_g_2 = w - NS.zeta.*res_smooth_w;
        u_g_1_active = ((abs(u_g_1) > NS.zeta.*NS.D) | (NS.D == 0));
        u_g_2_active = ((u_g_2 <= 0) | (u_g_2 >= NS.zeta));
        u_g_1_size = nnz(u_g_1_active);
        u_g_2_size = nnz(u_g_2_active);
        B_g_1 = zeros(n,1);
        B_g_1(u_g_1_active) = ones(u_g_1_size,1);
        B_g_2 = zeros(l,1);
        B_g_2(u_g_2_active) = ones(u_g_2_size,1);
        u_g_1_inactive = (B_g_1 == 0);
        u_g_2_inactive = (B_g_2 == 0);
        fprintf("l = %d, l_reduced = %d\n",l,nnz(u_g_2_inactive));
        % ___________________________________________________________________________________________________________________________ %      
        
        % =========================================================================================================================== %
        % Compute the right-hand-side and check the termination criterion
        % --------------------------------------------------------------------------------------------------------------------------- %
        prox_u_g_1 = u_g_1;
        prox_u_g_2 = u_g_2;
        prox_u_g_1 = max(abs(prox_u_g_1)-NS.zeta.*NS.D, zeros(n,1)).*sign(prox_u_g_1);  
        prox_u_g_2 = max(prox_u_g_2-NS.zeta, zeros(l,1)) + min(prox_u_g_2,zeros(l,1));  
        % SSN right-hand-side.
        rhs = [(1/NS.zeta).*([x;w]-[prox_u_g_1;prox_u_g_2]);              
               (-NS.C*x + w - NS.d -(1/NS.beta).*(v(1:l) - NS.v(1:l)));
               (NS.b - NS.A*x - (1/NS.beta).*(v(l+1:l+m) - NS.v(l+1:l+m)))];
      
        res_error = norm(rhs);
        if (res_error < tol)     % Stop if the desired accuracy is reached.
            break;
        end

        iter = iter + 1;
        if (pl > 1)
            if (iter == 1)
                fprintf(NS.fid,'___________________________________________________________________________________________________\n');
                fprintf(NS.fid,'___________________________________Semismooth Newton method________________________________________\n');
            end
            fprintf(NS.fid,'SSN Iteration                                         Residual Infeasibility                    \n');
            fprintf(NS.fid,'%4d                                                      %9.2e                      \n',iter,res_error);
        end
        % ___________________________________________________________________________________________________________________________ %

        % =========================================================================================================================== %
        % Check if we need to re-compute the factorization. If so compute it and store its factors in NS
        % --------------------------------------------------------------------------------------------------------------------------- %
        % update_factorization: true if the factorization must be recomputed  
        if (iter == 1 || ((nnz(NS.B_delta - B_delta) + nnz(NS.u_g_1_active - u_g_1_active) ...    
                          + nnz(NS.u_g_2_active - u_g_2_active) > max_outliers)))
            update_factorization = true;
        else
            update_factorization = false;
        end
        if (update_factorization)
            num_of_factor = num_of_factor + 1;
            NS.B_delta = B_delta;
            NS.u_g_1_active = u_g_1_active; NS.u_g_1_size = u_g_1_size;
            NS.u_g_2_active = u_g_2_active; NS.u_g_2_size = u_g_2_size;
            NS.Q_bar = NS.Q(:,u_g_1_active)';
            NS.A_tilde = NS.A(:,u_g_1_active);
            tmp_C = NS.C_tr(:,u_g_2_inactive)';
            NS.C_tilde = tmp_C(:,u_g_1_active);
            NS.H_tilde = NS.Q_bar(:,u_g_1_active) + (NS.beta+(1/NS.rho)).*speye(u_g_1_size) ...
                         - spdiags(NS.beta.*NS.B_delta(u_g_1_active),0,u_g_1_size,u_g_1_size);
            K = [-NS.H_tilde [NS.C_tilde' NS.A_tilde']; [NS.C_tilde;NS.A_tilde] (1/NS.beta).*speye(m+nnz(u_g_2_inactive))];
            [NS.L_f,NS.D_f,NS.pp_f] = ldl(K,0.1/(max(NS.beta,NS.rho)),'vector'); %Small pivots allowed, to avoid 2x2 pivots.
        end
        % ___________________________________________________________________________________________________________________________ %
        
        % =========================================================================================================================== %
        % Compute the reduced right-hand side backsolve the system and update the remaining variables.
        % --------------------------------------------------------------------------------------------------------------------------- %
        dx(u_g_1_inactive) = -(NS.zeta).*rhs([u_g_1_inactive;false(2*l+m,1)]);
        dw(u_g_2_inactive) = -w(u_g_2_inactive);
        dv([u_g_2_active;false(m,1)]) = -rhs([false(n,1);u_g_2_active;false(l+m,1)]);
        reduced_rhs = rhs([u_g_1_active;false(l,1);u_g_2_inactive;true(m,1)]);
        if (nnz(u_g_2_active))
            reduced_rhs = reduced_rhs + ...
                          [NS.Q_bar(:,u_g_1_inactive)*dx(u_g_1_inactive)+(NS.C(u_g_2_active,u_g_1_active))'*dv([u_g_2_active;false(m,1)]);
                           (-tmp_C(:,u_g_1_inactive)*dx(u_g_1_inactive)+dw(u_g_2_inactive));
                           (-NS.A(:,u_g_1_inactive)*dx(u_g_1_inactive))];
        else
            reduced_rhs = reduced_rhs + ...
                          [NS.Q_bar(:,u_g_1_inactive)*dx(u_g_1_inactive);
                           (-tmp_C(:,u_g_1_inactive)*dx(u_g_1_inactive)+dw(u_g_2_inactive));
                           (-NS.A(:,u_g_1_inactive)*dx(u_g_1_inactive))];
        end
        warn_stat = warning;
        warning('off','all');
        lhs = NS.L_f'\(NS.D_f\(NS.L_f\reduced_rhs(NS.pp_f)));
        if (nnz(isnan(lhs)) > 0 || nnz(isinf(lhs)) > 0)
            NS.beta = NS.beta*0.9;
            NS.rho = NS.rho*0.9;
            continue;
        end
        warning(warn_stat);
        lhs(NS.pp_f) = lhs; 
        dx(u_g_1_active) = lhs(1:u_g_1_size,1);
        dv([u_g_2_inactive;true(m,1)]) = lhs(u_g_1_size+1:end);
        tmp_vec = -(w-NS.d - NS.C*(x+dx) - (1/NS.beta).*(v(1:l)+dv(1:l)-NS.v(1:l)));
        dw(u_g_2_active) = tmp_vec(u_g_2_active);
        % ___________________________________________________________________________________________________________________________ %
       if (iter == 1)
           alpha = 0.995;
       else
            alpha = Nonsmooth_Line_Search(NS,x,w,v,dx,dw,dv,mu,delta);    
       end
        x = x + alpha.*dx;
        w = w + alpha.*dw;
        v = v + alpha.*dv;
    end
    tol_achieved = norm(rhs);
    if (pl > 1)
        fprintf(NS.fid,'___________________________________________________________________________________________________\n');
    end
end 
% *********************************************************************************************************************************** %
% END OF FILE.
% *********************************************************************************************************************************** %
