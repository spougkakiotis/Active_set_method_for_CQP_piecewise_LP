function NS = build_Newton_structure(Q,D,C,A,b,d,c_1,c_2,x,w,v,z,beta,rho,zeta,lb,ub,PMM_iter)
% ==================================================================================================================== %
% build_Newton_structure: Store all relevant information about the Newton system.
% -------------------------------------------------------------------------------------------------------------------- %
% NS = build_Newton_structure(Q,D,C,A,b,c_1,c_2,x,w,v,z,beta,rho,zeta,lb,ub,PMM_iter) returns a MATLAB 
%      struct that holds the relevant information of the Newton system, required for solving the step equations in
%      the SSN-PMM.
% 
% Author: Spyridon Pougkakiotis.
% ____________________________________________________________________________________________________________________ %
    % ================================================================================================================ %
    % Store all the relevant information required from the semismooth Newton's method.
    % ---------------------------------------------------------------------------------------------------------------- %
    NS = struct();
    NS.x = x;
    NS.w = w;
    NS.v = v;
    NS.z = z;
    NS.b = b;
    NS.d = d;
    NS.c_1 = c_1;
    NS.c_2 = c_2;
    NS.m = size(b,1);
    NS.n = size(c_1,1);
    NS.l = size(c_2,1);
    NS.beta = beta;
    NS.rho = rho;
    NS.zeta = zeta;
    NS.PMM_iter = PMM_iter;
    NS.lb = lb;
    NS.ub = ub;
    NS.A = A;
    NS.A_tr = A';
    NS.C = C;
    NS.C_tr = C';
    NS.D = D;
    NS.Q = Q;
    NS.Q_diag = spdiags(Q,0);
    % ________________________________________________________________________________________________________________ %
end
% ******************************************************************************************************************** %
% END OF FILE.
% ******************************************************************************************************************** %
