function [solution_statistics] = IP_PMM_Quantile_regression_problems(pb_name,parameters,tol,max_IPM_iter,printlevel,fid)
% ============================================================================================================================ %
% This function loads various portfolio selection problems and solves them using SSN_PMM.
% ---------------------------------------------------------------------------------------------------------------------------- %
% INPUT: 
% pb_name:          must contain the name of the regression problem of interest. Available choices: 
%                      1. -> 'abalone_scale.txt',   2. -> 'cadata.txt',             3. -> 'cpusmall_scale.txt',
%                      4. -> 'E2006.train',         5. -> 'space_ga_scale.txt'.
% parameters:     a MATLAB struct containing various parameters needed to build the problem
%                      1. -> .tau = "weight of the elastic penalty",    2. -> .lambda = "regularization parameter",
%                      3. -> .alpha = "quantile level",
%
% tol:              indicates the tolerance required (Default at 1e-4).
% max_IPM_iter:     specifies the maximum allowed number of IPM iterations (def. at 100).
% printlevel:       specifies the printing options. See the documentation of SSN-PMM for more. (def. at 1).
% fid:              specifies the file on which the algorithmic printing is done.
%
% OUTPUT: The output is given in the form of a struct, collecting various statistics from the run of the method:
%       solution_statistics.total_time  -> Time needed to solve all specified problems
%                          .total_PMM_iters -> Number of PMM iterations performed
%                          .total_SSN_iters -> Number of SSN iterations performed
%                          .tol -> the tolerance used
%                          .objective_value -> a scalar containing the optimal objective value
%                          .status -> an integer containing the termination status of PMM
%                          .solution_struct -> the solution of the last problem solved
%                          .problem_name -> contains the name of the problem solved
%
% Author: Spyridon Pougkakiotis, New Haven, CT, USA, 2022.
% ____________________________________________________________________________________________________________________________ %
    % ======================================================================================================================== %
    % Check input and fill. Include all data files.
    % ------------------------------------------------------------------------------------------------------------------------ %
    if (isempty(pb_name) || nargin < 1)
        error('Portfolio problem not specified.\n');
    elseif (~strcmp(pb_name,'abalone_scale.txt') && ~strcmp(pb_name,'cadata.txt') && ~strcmp(pb_name,'cpusmall_scale.txt')...
            && ~strcmp(pb_name,'E2006.train') && ~strcmp(pb_name,'space_ga_scale.txt'))
        error('Incorrect dataset name.\n');
    end
    if (nargin < 2 || isempty(parameters))          
        error('Parameter struct not specified.\n');
    end
    if (~isfield(parameters,"tau"))
        parameters.tau = 0.5;
    end
    if (~isfield(parameters,"alpha"))
        parameters.alpha = 0.8;
    end
    if (~isfield(parameters,"lambda"))
        parameters.stock_cap = 1e-2;
    end
    if (nargin < 3 || isempty(tol))          tol = 1e-4;         end
    if (nargin < 4 || isempty(max_IPM_iter)) max_IPM_iter = 100; end
    if (nargin < 5 || isempty(printlevel))   printlevel = 1;     end
    if (nargin < 6 || isempty(fid))          fid = 1;            end

    %The path on which all the netlib problems lie
    pb_path = './Problem_Data/LIBSVM/matlab/Regression_problem_data/'; 
    pb_path = strcat(pb_path,pb_name);
    [y,Xi] = libsvmread(pb_path);
    %Open the file to write the results
    fileID = fopen('./Output_files/Quantile_regression_SSN_PMM_results.txt','a+');
    % ________________________________________________________________________________________________________________________ %

    % ======================================================================================================================== %
    % Prepare struct that contains all relevant solution info.
    % ------------------------------------------------------------------------------------------------------------------------ %
    solution_statistics = struct();
    solution_statistics.solution_struct = struct();
    solution_statistics.total_IPM_iters = 0;
    solution_statistics.total_time = 0;
    solution_statistics.tol = tol;
    solution_statistics.objective_value = Inf;                     % To keep objective value.
    solution_statistics.status = -1;                               % To keep convergence status.
    solution_statistics.problem_name = pb_name;                    % To keep the name of the problem solved.
     % ________________________________________________________________________________________________________________________ %

    % ========================================================================================================================= %
    % Set up the problem matrices.
    % ------------------------------------------------------------------------------------------------------------------------- %
    [l,dim] = size(Xi);
    C = -(1/l).*[ones(l,1) Xi];
    C = sparse(C);
    n = 1+dim;
    Q_x_plus = [0 sparse(1,dim); sparse(dim,1) (parameters.lambda*(1-parameters.tau)).*speye(dim)];
    Q = [Q_x_plus -Q_x_plus sparse(n,2*l); -Q_x_plus Q_x_plus sparse(n,2*l); sparse(2*l,2*n+2*l)];
    obj_const_term = ((parameters.alpha - 1).*ones(l,1))'*((1/l).*y);
    tmp_c_plus = (parameters.alpha-1).*(ones(1,l)*C)' + (parameters.tau*parameters.lambda).*([0; ones(dim,1)]);
    tmp_c_minus = (1 - parameters.alpha).*(ones(1,l)*C)' + (parameters.tau*parameters.lambda).*([0; ones(dim,1)]);
    c = [tmp_c_plus; tmp_c_minus ; ones(l,1); zeros(l,1)];
    A = [C -C -speye(l) speye(l)];
    b = -((1/l).*y);
    free_variables = [];
    pc = true;
    % _________________________________________________________________________________________________________________________ %
    
    tStart = tic;
    [x,v,z,opt,iter] = IP_PMM(c,A,Q,b,free_variables,tol,max_IPM_iter,pc,printlevel);
    tEnd = toc(tStart);
    solution_statistics.total_IPM_iters = iter;
    solution_statistics.total_time = tEnd;  
    solution_statistics.objective_value = c'*x + (1/2).*(x'*(Q*x)) +  obj_const_term;
    solution_statistics.solution_struct.x = x;
    solution_statistics.solution_struct.v = v;
    solution_statistics.solution_struct.z = z;

    if (opt == 1)                                       % Success
       fprintf(fileID,'The optimal solution objective is %d.\n',solution_statistics.objective_value);
    else                                                                % Reached maximum iterations
       fprintf(fileID,'Maximum number of iterations reached.\n Returning the last iterate.\n'); 
    end
    fprintf(fileID,['Name = %s & IPM iters = %d & Time = %.2e & opt = %s  \n'],pb_name, iter,tEnd, string(opt == 1)); 
    solution_statistics.status = opt;
    fprintf(fileID,'The total IPM iterates were: %d  and the total time was %d.\n',iter,solution_statistics.total_time);
    fclose(fileID);
end
