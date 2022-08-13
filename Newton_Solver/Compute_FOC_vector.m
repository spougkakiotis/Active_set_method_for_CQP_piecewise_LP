function M_hat = Compute_FOC_vector(NS,x,w,v)
% ========================================================================================= %
% M_hat = Compute_FOC_vector(NS,x,w,v)
% ----------------------------------------------------------------------------------------- %
% This function takes as an input the Newton structure 
% containing information necessary to build the SSN-PMM 
% subproblems and returns the smoothed first-order 
% optimality conditions (FOC) evaluated at (x,w,v,NS.z).
% _________________________________________________________________________________________ %
    % ===================================================================================== %
    % Compute the projection onto K.
    % ------------------------------------------------------------------------------------- %
    proj_vec = (1/NS.beta).*(NS.z) + x;  
    tmp_lb = (proj_vec < NS.lb);
    tmp_ub = (proj_vec > NS.ub);
    proj_vec(tmp_lb) = NS.lb(tmp_lb);
    proj_vec(tmp_ub) = NS.ub(tmp_ub);
    % _____________________________________________________________________________________ %
    
    % ===================================================================================== %
    % Compute the natural primal residual by evaluating the
    % prox of g_1(x) = \|Dx\|_1, g_2(x) = sum max{w,0}.
    % ------------------------------------------------------------------------------------- %
    res_smooth_x = NS.c_1 + NS.Q*x - [NS.C_tr NS.A_tr]*v+ ... 
                   NS.z + NS.beta.*x - NS.beta.*proj_vec + (1/NS.rho).*(x-NS.x);
    res_smooth_w = NS.c_2 + v(1:NS.l);
    u_g_1 = x - NS.zeta.*res_smooth_x;
    u_g_2 = w - NS.zeta.*res_smooth_w;    
    prox_u_g_1 = u_g_1;
    prox_u_g_1 = max(abs(prox_u_g_1)-NS.zeta.*NS.D, zeros(NS.n,1)).*sign(prox_u_g_1);  
    prox_u_g_2 = u_g_2;
    prox_u_g_2 = max(prox_u_g_2-NS.zeta, zeros(NS.l,1)) + min(prox_u_g_2,zeros(NS.l,1));
    % _____________________________________________________________________________________ %
    
    % ===================================================================================== %
    % Evaluate the natural residual at (x,w,v).
    % ------------------------------------------------------------------------------------- %
    M_hat = [(1/NS.zeta).*([x;w] - [prox_u_g_1;prox_u_g_2]); 
             (-NS.C*x + w - NS.d -(1/NS.beta).*(v(1:NS.l) - NS.v(1:NS.l)));
             (NS.b - NS.A*x - (1/NS.beta).*(v(NS.l+1:NS.l+NS.m) - NS.v(NS.l+1:NS.l+NS.m)))];
    % _____________________________________________________________________________________ %
end

