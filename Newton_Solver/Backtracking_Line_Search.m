function alpha = Backtracking_Line_Search(NS,x,v,dx,dv,mu,delta)
% ==================================================================================================================== %
% Backtracking Line Search
% -------------------------------------------------------------------------------------------------------------------- %
% alpha = Backtracking_Line_Search(NS,start_point,direction,mu,delta,method)
%                                                    takes as an input the Newton structure (NS), containing all 
%                                                    relevant information needed to build the augmented Lagrangian, 
%                                                    x, which is the point upon which the Lagrangian is evaluated 
%                                                    (as well as its gradient), dx, which is the computed Newton 
%                                                    direction, mu, which is the fraction of decrease in the 
%                                                    Lagrangian, predicted by linear extrapolation, that we 
%                                                    are willing to accept, and finally, delta, which is the 
%                                                    maximum possible step-length.
%                                                    It returns the final step-length which satisfies the line-search
%                                                    requirements.
%                                                           
% Author: Spyridon Pougkakiotis.
% ____________________________________________________________________________________________________________________ %

    % ================================================================================================================ %
    % j-th SSN iteration: 
    % Compute and store phi(x_j) = c^T x_j + y_k^T(Ax_j) + (beta_k/2)||Ax_j - b||^2 -
    %                              x_j^T(Proj_{K^*}(z_k - beta_k x_j) + (1/(2rho_k))||x_j - x_k||^2
    %                              -(1/(2beta_k))||Proj_{K^*}(z_k - beta_k x_j) - z_k||^2
    % as well as \nabla (phi(x_j)) = c + A^Ty_k + beta_k A^T(Ax_j-b) - Proj_{K^*}(z_k - beta_k x_j) + (1/rho_k)(x_j-x_k)
    % ---------------------------------------------------------------------------------------------------------------- %
    direction = [dx;dv];
    w = (1/NS.beta).*(NS.z) + x;        
    w_lb = (w < NS.lb);
    w_ub = (w > NS.ub);
    w(w_lb) = NS.lb(w_lb);
    w(w_ub) = NS.ub(w_ub);
    pr_res = NS.A*x-NS.b;
    phi =  NS.c'*x + (1/2)*(x'*(NS.Q*x)) - NS.v'*pr_res + (NS.beta/2)*(norm(pr_res,2))^2 ...
            + (1/(2*NS.beta))*(-norm(NS.z)^2+ norm(NS.z+NS.beta.*x - NS.beta.*w)^2) ...
            + (1/(2*NS.rho))*norm(x-NS.x)^2 + (NS.nu*NS.beta/2)*(norm(pr_res + (1/NS.beta).*(v-NS.v)))^2;
    nabla_phi = [NS.c + NS.Q*x + NS.A_tr*((1+NS.nu).*(-NS.v + (NS.beta).*(pr_res)) + v) ...
                 + NS.z+NS.beta.*x- NS.beta.*w  + (1/NS.rho).*(x-NS.x); 
                 NS.nu.*(NS.A*x + (1/NS.beta).*(v - NS.v) - NS.b)];
    % ________________________________________________________________________________________________________________ %
    
    % ================================================================================================================ %
    % Let alpha = delta, and evaluate phi(x + alpha.*dx).
    % ---------------------------------------------------------------------------------------------------------------- %
    alpha = delta;
    x_new = x + alpha.*dx;
    v_new = v + alpha.*dv;
    pr_res = NS.A*x_new-NS.b;
    w = (1/NS.beta).*(NS.z) + x_new; 
    w_lb = (w < NS.lb);
    w_ub = (w > NS.ub);
    w(w_lb) = NS.lb(w_lb);
    w(w_ub) = NS.ub(w_ub);
    phi_new =  NS.c'*x_new + (1/2)*(x_new'*(NS.Q*x_new)) - NS.v'*pr_res + (NS.beta/2)*(norm(pr_res,2))^2 ...
               + (1/(2*NS.beta))*(-norm(NS.z)^2+ norm(NS.z+NS.beta.*x_new - NS.beta.*w)^2) ...
               + (1/(2*NS.rho))*norm(x_new-NS.x)^2 + ((NS.nu*NS.beta)/2)*(norm(pr_res + (1/NS.beta).*(v_new-NS.v)))^2;
    % ________________________________________________________________________________________________________________ %
    
    % ================================================================================================================ %
    % Iterate until you find a step-length satisfying the Armijo-Goldstein condition.
    % ---------------------------------------------------------------------------------------------------------------- %
    counter = 1;
    while (phi_new > phi + mu*(alpha)*(nabla_phi'*direction))
        counter = counter + 4;
        alpha = delta^counter;
        x_new = x + alpha.*dx;
        v_new = v + alpha.*dv;
        pr_res = NS.A*x_new-NS.b;
        w = (1/NS.beta).*(NS.z) + x_new;   
        w_lb = (w < NS.lb);
        w_ub = (w > NS.ub);
        w(w_lb) = NS.lb(w_lb);
        w(w_ub) = NS.ub(w_ub);
        phi_new =  NS.c'*x_new + (1/2)*(x_new'*(NS.Q*x_new)) - NS.v'*pr_res + (NS.beta/2)*(norm(pr_res,2))^2 ...
                   + (1/(2*NS.beta))*(-norm(NS.z)^2+ norm(NS.z+NS.beta.*x_new - NS.beta.*w)^2) ...
                   + (1/(2*NS.rho))*norm(x_new-NS.x)^2 ...
                   + ((NS.nu*NS.beta)/2)*(norm(pr_res + (1/NS.beta).*(v_new-NS.v)))^2;
    end
    % ________________________________________________________________________________________________________________ %
end
% ******************************************************************************************************************** %
% END OF FILE.
% ******************************************************************************************************************** %

