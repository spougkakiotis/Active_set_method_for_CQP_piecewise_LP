function [res_p,res_d,compl] = compute_residual(Q,D,C,C_tr,A,A_tr,b,d,c_1,c_2,lb,ub,x,w,v,z)
% ==================================================================================================================== %
% This function takes the QP problem data as well as the current iterate as input, and outputs the 
% primal, dual infeasibilities, as well as complementarity.
% -------------------------------------------------------------------------------------------------------------------- %
    n = size(c_1,1);  l = size(c_2,1);
    res_p = [(w - d- C*x);(b - A*x)];                                                     % Primal residual.
    temp_res_d_x = x - c_1 - Q*x + [C_tr A_tr]*v -z;
    temp_res_d_x = max(abs(temp_res_d_x)-D, zeros(n,1)).*sign(temp_res_d_x);               % Prox of\|Dx\|_1.
    temp_res_d_w = w - c_2 - v(1:l);
    temp_res_d_w = max(temp_res_d_w-ones(l,1),zeros(l,1)) + min(temp_res_d_w,zeros(l,1));  % Prox of sum max{w,0}.
    res_d = [(x - temp_res_d_x); (w - temp_res_d_w)];                                      % Dual residual.
    temp_compl = x + z;
    temp_lb = (temp_compl < lb);
    temp_ub = (temp_compl > ub);
    temp_compl(temp_lb) = lb(temp_lb);
    temp_compl(temp_ub) = ub(temp_ub);
    compl = norm(x -  temp_compl);                                      % Measure of the complementarity between x and z.
% ____________________________________________________________________________________________________________________ %
end
% ******************************************************************************************************************** %
% END OF FILE.
% ******************************************************************************************************************** %
