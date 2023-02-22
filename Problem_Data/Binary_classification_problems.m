function [solution_statistics] = Binary_classification_problems(pb_name,parameters,tol,max_PMM_iter,printlevel,fid,warm_start)
% ============================================================================================================================ %
% This function loads various portfolio selection problems and solves them using SSN_PMM.
% ---------------------------------------------------------------------------------------------------------------------------- %
% INPUT: 
% pb_name:          must contain the name of the regression problem of interest. Problem options: 
%           1. -> 'real-sim',   2. -> 'rcv1_train.binary',             3. -> 'news20.binary',
%           4. -> 'gisette_scale',         5. -> 'epsilon_normalized'.
%
% parameters:     a MATLAB struct containing various parameters needed to build the problem
%                      1. -> .l1 = "true if L1-SVM false if L2-SVM",    2. -> .lambda = "regularization parameter",
%
% tol:              indicates the tolerance required (Default at 1e-4).
% max_PMM_iter:     specifies the maximum allowed number of PMM iterations (def. at 100).
% printlevel:       specifies the printing options. See the documentation of SSN-PMM for more. (def. at 1).
% fid:              specifies the file on which the algorithmic printing is done.
% warm_start        "matrix-free" for matrix-free ADMM, "standard" for standard ADMM.
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
        error('Classification problem not specified.');
    elseif (~strcmp(pb_name,'real-sim') && ~strcmp(pb_name,'rcv1_train.binary') && ~strcmp(pb_name,'news20.binary')...
            && ~strcmp(pb_name,'gisette_scale') && ~strcmp(pb_name,'epsilon_normalized') && ~strcmp(pb_name,'breast-cancer_scale.txt')...
            && ~strcmp(pb_name,'a9a.txt'))
        error('Incorrect dataset name.');
    end
    if (nargin < 2 || isempty(parameters))          
        error('Parameter struct not specified.');
    end
    if (~isfield(parameters,"lambda"))
        parameters.lambda = 1e-2;
    end
    if (nargin < 3 || isempty(tol))          tol = 1e-4;         end
    if (nargin < 4 || isempty(max_PMM_iter)) max_PMM_iter = 100; end
    if (nargin < 5 || isempty(printlevel))   printlevel = 1;     end
    if (nargin < 6 || isempty(fid))          fid = 1;            end

    %The path on which all the netlib problems lie
    pb_path = './Problem_Data/LIBSVM/matlab/Binary_classification_data/'; 
    pb_path = strcat(pb_path,pb_name);
    [y,Xi] = libsvmread(pb_path);
    %Open the file to write the results
    fileID = fopen('./Output_files/Binary_classification_SSN_PMM_results.txt','a+');
    % ________________________________________________________________________________________________________________________ %

    % ======================================================================================================================== %
    % Prepare struct that contains all relevant solution info.
    % ------------------------------------------------------------------------------------------------------------------------ %
    solution_statistics = struct();
    solution_statistics.total_PMM_iters = 0;
    solution_statistics.total_time = 0;
    solution_statistics.tol = tol;
    solution_statistics.total_SSN_iters = 0;
    solution_statistics.objective_value = Inf;                     % To keep objective value.
    solution_statistics.status = -1;                               % To keep convergence status.
    solution_statistics.total_num_of_factorizations = 0;           % To keep the maximum number of factorizations performed.
    solution_statistics.problem_name = pb_name;                    % To keep the name of the problem solved.
     % ________________________________________________________________________________________________________________________ %

    % ========================================================================================================================= %
    % Set up the problem matrices.
    % ------------------------------------------------------------------------------------------------------------------------- %
    [l,dim] = size(Xi);
    d = (1/l).*ones(l,1);
    C = -(1/l).*[-y.*ones(l,1) y.*Xi];
    C = sparse(C);
    
    A = sparse(0,1+dim);
    b = zeros(0,1);
    n = 1+dim;
    if (parameters.regularizer == "l1")
        D = [0;parameters.lambda.*ones(dim,1)];
        Q = sparse(n,n);
    elseif (parameters.regularizer == "l2")
        Q = [0 sparse(1,dim); sparse(dim,1) (parameters.lambda.*speye(dim))];
        D = zeros(n,1);
    else
        Q = [0 sparse(1,dim); sparse(dim,1) (parameters.tau_2*parameters.lambda).*speye(dim)];
        D = [0;(parameters.tau_1*parameters.lambda).*ones(dim,1)];
    end
    c_1 = zeros(n,1);
    c_2 = zeros(l,1);
    lb = -Inf.*ones(n,1);
    ub = Inf.*ones(n,1);
    % _________________________________________________________________________________________________________________________ %
        
    tStart = tic;
    [solution_struct] = SSN_PMM(Q,D,A,C,b,d,c_1,c_2,lb,ub,tol,max_PMM_iter,printlevel,fid,warm_start);
    tEnd = toc(tStart);
        
    solution_statistics.total_PMM_iters = solution_struct.PMM_iter;
    solution_statistics.total_time = tEnd;  
    solution_statistics.total_SSN_iters = solution_struct.SSN_iter;
    solution_statistics.total_num_of_factorizations = solution_struct.total_num_of_factorizations;
    solution_statistics.objective_value = solution_struct.obj_val;
    if (solution_struct.opt == 1)                                       % Success
       fprintf(fileID,'The optimal solution objective is %d.\n',solution_struct.obj_val);
    else                                                                % Reached maximum iterations
       fprintf(fileID,['Maximum number of iterations reached.\n',...
                       'Returning the last iterate.\n']); 
    end
    fprintf(fileID,['Name = %s & PMM iters = %d & SSN iters = %d & num. of factorizations = %.2e & ' ...
            'Time = %.2e & opt = %s  \n'],pb_name, solution_struct.PMM_iter, solution_struct.SSN_iter,...
                                         solution_struct.total_num_of_factorizations, tEnd, string(solution_struct.opt == 1)); 
    solution_statistics.status = solution_struct.opt;
    solution_statistics.solution_struct= solution_struct;
    fprintf(fileID,'The total PMM iterates were: %d (with %d total SSN iterates) and the total time was %d.\n',...
                    solution_statistics.total_PMM_iters,solution_statistics.total_SSN_iters,solution_statistics.total_time);
    fclose(fileID);
end

