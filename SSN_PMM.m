function [solution_struct] = SSN_PMM(Q, D, A, C, b, d, c_1, c_2, lb, ub, tol, maxit, printlevel, print_fid, warm_start)
% ======================================================================================================================== %
% This function is a primal-dual Semismooth Newton Proximal Method of Multipliers, suitable for solving convex quadratic 
% programming problems. The method takes as input a problem of the following form:
%
%                            min   (1/2)(x)^T Q (x) + c_1^T x + c_2^T w + \sum_{j=1}^l max(w_j,0) + \|Dx\|_1,
%                            s.t.  A x = b,
%                                  C x + d - w = 0,
%                                  lb <= x <= ub,
%
% where D is a diagonal positive semi-definite matrix, and solves it to epsilon-optimality, 
% returning the primal and dual optimal solutions (or a message indicating that the optimal solution was not found). 
%
% INPUT PARAMETERS:
% SSN_PMM(Q, D, A, C, b, d, c_1, c_2, lb, ub, tol, maxit, printlevel):  
%
%                            Q -> smooth Hessian matrix,
%                            D -> diagonal vector representing a positive semi-definite diagonal matrix,
%                            A -> constraint matrix, 
%                            C -> linear function appearing within max{,0},
%                            b -> equality constraints' right-hand-side,
%                            d -> linear shift appearing within max{,0},
%                            c_1 -> linear part of the objective (corresponding to x),
%                            c_2 -> linear part of the objective (corresponding to w),
%                            lb -> vector containing the lower-bound restrictions of x (unbounded default),
%                            ub -> vector containing the upper-bound restrictions of x (unbounded default),
%                            tol -> error tolerance (10^(-4) default),
%                            maxit -> maximum number of PMM iterations (200 default),
%                            printlevel -> 0 for not printing intermediate iterates,
%                                       -> 1 for printing only PMM iterates (default),
%                                       -> 2 for additionally printing SNN iterates.
%                            printf_fid -> file ID to print output,
%                            warm_start -> "matrix-free" for matrix-free ADMM
%                                       -> "standard" for standard ADMM.
%
% OUTPUT: [solution_struct], where, the struct contains the following entries
%         x: Optimal primal solution
%         w: Optimal primal solution
%         v: Lagrange multiplier vector corresponding to equality constraints
%         z: Lagrange multiplier vector corresponding to box constraints on x
%         opt: 0, if the maximum number of iterations is reached,
%              1, if the tol-optimal solution was found,
%              2, if the method terminated due to numerical inaccuracy.
%         PMM_iter: number of PMM iterations to termination.
%         SSN_iter: number of SSN iterations to termination.
%         total_num_of_factorizations: the total number of factorizations performed.
%
% Author: Spyridon Pougkakiotis, October 2022, New Haven, CT, USA.
% ________________________________________________________________________________________________________________________ %
    % ==================================================================================================================== %
    % Parameter filling and dimensionality testing.
    % -------------------------------------------------------------------------------------------------------------------- %
    if (~issparse(A))                           % Ensure that A is sparse.
        A = sparse(A);
    end
    if (~issparse(C))
        C = sparse(C);
    end
    if (~issparse(Q))                           % Ensure that Q is sparse.
        Q = sparse(Q);
    end
  
    if (issparse(b))    b = full(b);       end  % Make sure that b, d, c_1, c_2 are full.
    if (issparse(c_1))  c_1 = full(c_1);   end
    if (issparse(c_2))  c_1 = full(c_2);   end
    if (issparse(d))    d = full(d);       end

    % Make sure that b and c are column vectors of dimension m and n.
    if (size(b,2) > 1)   b = (b)';     end
    if (size(c_1,2) > 1) c_1 = (c_1)'; end
    if (size(c_2,2) > 1) c_2 = (c_2)'; end
    if (size(d,2) > 1)   d = (d)';     end

    m = size(b,1);  n = size(c_1,1);   l = size(c_2,1);
    if (~isequal(size(c_1,1),n) || ~isequal(size(b,1),m) || ~isequal(size(c_2,1),l) || ~isequal(size(d,1),l))
        error('problem dimensions incorrect');
    end
    if (isempty(D))
        D = zeros(n,1);
    elseif (size(D) == size(Q))
        D = spdiags(D,0);
    elseif (size(D) ~= size(c_1))
        error('Vector D representing the non-smooth Hessian is given incorrectly.'\n);
    end
    % ===================================================================================================================== %
    % If some of the two constraints are null, put zero constraints in place.
    % --------------------------------------------------------------------------------------------------------------------- %
    if (m == 0)
        m = 1; b = 0; A = sparse(1,n);
    elseif (l == 0)
        l = 1; d = 0; C = sparse(1,n);
    end
    % _____________________________________________________________________________________________________________________ %

    if (~isequal(size(C),[l,n]))
        error('Matrix C representing the linear function within the max function has incorrect dimensions.'\n);        
    end
    % Set default values for missing parameters.
    if (nargin < 9  || (isempty(lb)))           lb = -Inf.*ones(n,1);     end
    if (nargin < 10 || (isempty(ub)))           ub = Inf.*ones(n,1);      end
    if (nargin < 11 || isempty(tol))            tol = 1e-4;               end
    if (nargin < 12 || isempty(maxit))          maxit = 200;              end
    if (nargin < 13 || isempty(printlevel))     printlevel = 1;           end
    if (nargin < 14 || isempty(print_fid))      print_fid = 1;            end
    if (nargin < 15 || isempty(warm_start))     warm_start = "standard";  end
    pl = printlevel;
    % ____________________________________________________________________________________________________________________ %
    % Store the transpose constraint matrices for computational efficiency.
    A_tr = A';                           
    C_tr = C';
    % ==================================================================================================================== %
    % Initialization - Starting Point:
    % Choose an initial starting point (x,v,z) such that any positive variable is set to a positive constant and 
    % free variables are set to zero.
    % -------------------------------------------------------------------------------------------------------------------- %
    beta = 1e2;   rho = 5e2;   zeta = 1;                                % Initial primal and dual regularization values.
    warm_start_tol = 1e-3;
    if (warm_start == "matrix-free")
        warm_start_maxit = 100;
        [x,w,v,z,ws_opt] = ChamPock_warmstart(Q,D,A,A_tr,C,C_tr,b,d,c_1,c_2,lb,ub,warm_start_tol,warm_start_maxit);
    else
        warm_start_maxit = 100;
        [x,w,v,z,ws_opt,~] = ADMM_warmstart(Q,D,A,A_tr,C,C_tr,b,d,c_1,c_2,lb,ub,warm_start_tol,warm_start_maxit);
    end
    % ____________________________________________________________________________________________________________________ %

    % ==================================================================================================================== %  
    % Initialize parameters
    % -------------------------------------------------------------------------------------------------------------------- %
    max_SSN_iters = 4000;                                               % Number of maximum SSN iterations. 
    PMM_iter = 0;                                                       % Monitors the number of PMM iterations.
    SSN_iter = 0;                                                       % Monitors the number of SSN iterations.
    total_num_of_factorizations = 0;                                    % Monitors the number of factorizations.
    in_tol_thr = tol;                                                   % Inner tolerance for Semismooth Newton method.
    SSN_maxit = 20;                                                      % Maximum number of SSN iterations.
    opt = 0;                                                            % Variable monitoring the optimality.
    PMM_header(print_fid,pl);                                           % Set the printing choice.
    reg_limit = 1e+6;                                                   % Maximum value for the penalty parameters.
    solution_struct = struct();                                         % Struct to contain output information.
    [res_p,res_d,compl] = compute_residual(Q,D,C,C_tr,A,A_tr,b,d,c_1,c_2,lb,ub,x,w,v,z);
    % ____________________________________________________________________________________________________________________ %

    while (PMM_iter < maxit)
    % -------------------------------------------------------------------------------------------------------------------- %
    % SSN-PMM Main Loop structure:
    % Until (primal feasibility < tol && dual feasibility < tol && complementarity < tol) do
    %   Call Semismooth Newton's method to approximately minimize the primal-dual augmented Lagrangian w.r.t. x,w,v;
    %   update z;
    %   update the reuglarization paramters;
    %   k = k + 1;
    % End
    % -------------------------------------------------------------------------------------------------------------------- %
        % ================================================================================================================ %
        % Check termination criteria
        % ---------------------------------------------------------------------------------------------------------------- %
        if ((PMM_iter == 0) && ws_opt && (warm_start_tol <= tol))
            fprintf('optimal solution found\n');
            opt = 1;
            break;
        elseif (norm(res_p,inf)/(1+norm([d;b],inf)) < tol && compl/(1 + norm(x,inf) + norm(z,inf)) < tol && ...
                norm(res_d,inf)/(1+norm(c_1,inf)+norm(x,inf)+norm(c_2,inf)+norm(w,inf)) < tol)
            fprintf('optimal solution found\n');
            opt = 1;
            break;
        end
        PMM_iter = PMM_iter+1;
        % ________________________________________________________________________________________________________________ %
                
        % ================================================================================================================ %
        % Build or update the Newton structure
        % ---------------------------------------------------------------------------------------------------------------- %
        if (PMM_iter == 1) 
            NS = build_Newton_structure(Q,D,C,A,b,d,c_1,c_2,x,w,v,z,beta,rho,zeta,lb,ub,PMM_iter);
            NS.fid = print_fid;
        else
            NS.x = x; NS.w = w; NS.v = v; NS.z = z; NS.beta = beta; NS.rho = rho; NS.PMM_iter = PMM_iter; NS.zeta = zeta;
        end
        % ________________________________________________________________________________________________________________ %
        
        % ================================================================================================================ %
        % Call semismooth Newton method to find the x-update.
        % ---------------------------------------------------------------------------------------------------------------- %
        res_vec = [0.1*norm(res_p); 0.1*norm(res_d); 1];
        in_tol = max(min(res_vec),in_tol_thr);
        SSN_tol_achieved = 2*max(res_vec);
        counter = 0; 
        while (SSN_tol_achieved > max(1e-1*max(res_vec),min(res_vec))) % && counter < 1
            counter = counter + 1;
            [x, w, v, SSN_in_iters, SSN_tol_achieved, num_of_fact] = Semismooth_Newton(NS,in_tol,SSN_maxit,pl);
            total_num_of_factorizations = num_of_fact + total_num_of_factorizations;
            SSN_iter = SSN_iter + SSN_in_iters;
            NS.x = x;
            NS.w = w;
            NS.v = v;
            if (SSN_iter >= max_SSN_iters)
                fprintf('Maximum number of inner iterations is reached. Terminating without optimality.\n');
                tmp_vec = (1/beta).*(NS.z) + x;
                tmp_lb = find(tmp_vec < lb);
                tmp_ub = find(tmp_vec > ub);
                tmp_vec(tmp_lb) = lb(tmp_lb);
                tmp_vec(tmp_ub) = ub(tmp_ub);
                z = NS.z + beta.*x - beta.*tmp_vec;
                break;
            end
        end      
        % ________________________________________________________________________________________________________________ %
        
        % ================================================================================================================ %
        % Perform the dual update (z).
        % ---------------------------------------------------------------------------------------------------------------- %
        tmp_vec = (1/beta).*(NS.z) + x;
        tmp_lb = find(tmp_vec < lb);
        tmp_ub = find(tmp_vec > ub);
        tmp_vec(tmp_lb) = lb(tmp_lb);
        tmp_vec(tmp_ub) = ub(tmp_ub);
        z = NS.z + beta.*x - beta.*tmp_vec;
        [new_res_p,new_res_d,compl] = compute_residual(Q,D,C,C_tr,A,A_tr,b,d,c_1,c_2,lb,ub,x,w,v,z);
        % ________________________________________________________________________________________________________________ %
        
        % ================================================================================================================ %
        % If the overall primal and dual residual error is decreased, 
        % we increase the penalty parameters aggressively.
        % If not, we continue increasing the penalty parameters slower, limiting the increase to the value 
        % of the regularization threshold.
        % ---------------------------------------------------------------------------------------------------------------- %
        [beta,rho] = update_PMM_parameters(res_p,res_d,new_res_p,new_res_d,beta,rho,reg_limit);
        res_p = new_res_p;
        res_d = new_res_d;
        % ________________________________________________________________________________________________________________ %

        % ================================================================================================================ %
        % Print iteration output.  
        % ---------------------------------------------------------------------------------------------------------------- %
        pres_inf = norm(res_p,inf);
        dres_inf = norm(res_d,inf);  
        PMM_output(print_fid,pl,PMM_iter,SSN_iter,pres_inf,dres_inf,compl,SSN_tol_achieved,beta,rho);
        % ________________________________________________________________________________________________________________ %
    end % while (iter < maxit)
    % ==================================================================================================================== %  
    % The PMM has terminated. Print results, and prepare output.
    % -------------------------------------------------------------------------------------------------------------------- %
    fprintf(print_fid,'outer iterations: %5d\n', PMM_iter);
    fprintf(print_fid,'inner iterations: %5d\n', SSN_iter);
    [res_p,res_d,compl] = compute_residual(Q,D,C,C_tr,A,A_tr,b,d,c_1,c_2,lb,ub,x,w,v,z);
    fprintf(print_fid,'primal feasibility: %8.2e\n', norm(res_p,inf));
    fprintf(print_fid,'dual feasibility: %8.2e\n', norm(res_d,inf));
    fprintf(print_fid,'complementarity: %8.2e\n', compl);  
    fprintf(print_fid,'total number of factorizations: %5d\n', total_num_of_factorizations);
    solution_struct.x = x;  solution_struct.w = w; solution_struct.v = v;  solution_struct.z = z;
    solution_struct.opt = opt;  
    solution_struct.PMM_iter = PMM_iter;
    solution_struct.SSN_iter = SSN_iter;    
    solution_struct.total_num_of_factorizations = total_num_of_factorizations;
    solution_struct.obj_val = c_1'*x + c_2'*w + (1/2)*(x'*(Q*x)) + sum(max(w,zeros(l,1))) + norm(D.*x,1);
    % ____________________________________________________________________________________________________________________ %
end
% ************************************************************************************************************************ %
% END OF FILE
% ************************************************************************************************************************ %