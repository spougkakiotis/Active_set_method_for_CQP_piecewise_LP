function [solution_statistics] = ADMM_Portfolio_optimization_problems(pb_name,risk_measure,tol,max_ADMM_iter,printlevel,fid)
% ============================================================================================================================ %
% This function loads various portfolio selection problems and solves them using SSN_PMM.
% ---------------------------------------------------------------------------------------------------------------------------- %
% INPUT: 
% pb_name:          must contain the name of the portfolio
%                   optimization problem dataset of interest. Available choices: 
%                      1. -> "DowJones",   2. -> "FF49Industries", 3. -> "FTSE100",
%                      4. -> "NASDAQ100",  5. -> "NASDAQComp",     6. -> "SP500".
% risk_measure:     a MATLAB struct containing the name of the risk measure the user
%                   wants to minimize (names are case sensitive):
%                      1. -> .name = "CVaR",    2. -> .name = "MAD",    3. -> .name = "Variance",
%                   as well as any associated parameters. 
%                       For CVaR we need:
%                           1. -> .alpha = percentile (def. 0.01) 2. -> .tau = ell-1 regularization parameter (def. 1e-1),
%                           3. -> .stock_cap = maximum percentage of wealth allowed in a single stock (def. 0.4),
%                           4. -> .short_cap = maximum percentage of wealth allowed in shorting a stock (def. -0.1).
%                       For MAsD we need:
%                           1. -> .tau = ell-1 regularization parameter (def. 1e-1),
%                           2. -> .stock_cap = maximum percentage of wealth allowed in a single stock (def. 0.4).                           ...
%                           3. -> .short_cap = maximum percentage of wealth allowed in shorting a stock (def. -0.1).
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
        error('Portfolio problem not specified.\n');
    elseif (~strcmp(pb_name,"DowJones") && ~strcmp(pb_name,"FF49Industries") && ~strcmp(pb_name,"FTSE100")...
            && ~strcmp(pb_name,"NASDAQ100") && ~strcmp(pb_name,"NASDAQComp") && ~strcmp(pb_name,"SP500"))
        error('Incorrect dataset name.\n');
    end
    if (nargin < 2 || isempty(risk_measure))          
        error('Risk measure not specified.\n');
    elseif (~strcmp(risk_measure.name,"CVaR") && ~strcmp(risk_measure.name,"MAsD") && ~strcmp(risk_measure.name,"Variance"))
        error('Incorrect risk measure name.\n');
    end
    if (strcmp(risk_measure.name,"CVaR") || strcmp(risk_measure.name,"MAsD"))
        if (strcmp(risk_measure.name,"CVaR"))
            if (~isfield(risk_measure,"alpha"))
                risk_measure.alpha = 0.1;
            end
        end
        if (~isfield(risk_measure,"tau"))
            risk_measure.tau = 0.1;
        end
        if (~isfield(risk_measure,"stock_cap"))
            risk_measure.stock_cap = 0.4;
        end
        if (~isfield(risk_measure,"short_cap"))
            risk_measure.stock_cap = -0.1;
        end
    end 
    if (nargin < 3 || isempty(tol))          tol = 1e-4;         end
    if (nargin < 4 || isempty(max_ADMM_iter)) max_ADMM_iter = 100; end
    if (nargin < 5 || isempty(printlevel))   printlevel = 1;     end
    if (nargin < 6 || isempty(fid))          fid = 1;            end

    %The path on which all the netlib problems lie
    pb_path = './Problem_Data/Portfolio_optimization_data/Datasets/'; 
    pb_path = strcat(pb_path,pb_name);
    %Finds all the Netlib problems and stores their names in a struct
    dirct = dir(fullfile(pb_path,'*.mat'));
    %Open the file to write the results
    fileID = fopen('./Output_files/Portfolio_opt_SSN_PMM_results.txt','a+');
    % ________________________________________________________________________________________________________________________ %

    % ======================================================================================================================== %
    % Prepare struct that contains all relevant solution info.
    % ------------------------------------------------------------------------------------------------------------------------ %
    solution_statistics = struct();
    solution_statistics.solution_struct = struct();
    solution_statistics.total_ADMM_iters = 0;
    solution_statistics.total_time = 0;
    solution_statistics.tol = tol;
    solution_statistics.objective_value = Inf;                     % To keep objective value.
    solution_statistics.status = -1;                               % To keep convergence status.
    solution_statistics.problem_name = pb_name;                    % To keep the name of the problem solved.
     % ________________________________________________________________________________________________________________________ %
   
    % Load the problem struct: fields  = {'Index_Returns', 'Assets_Returns'}; 
    pb_struct = load(fullfile(pb_path,dirct.name)); 
    
    % ========================================================================================================================= %
    % Set up the basic constraints for the single-period mean-risk portfolio optimization problem, and gather data matrices.
    % ------------------------------------------------------------------------------------------------------------------------- %
    [l,n] = size(pb_struct.Assets_Returns);
    benchmark_mean_return = mean(pb_struct.Index_Returns);
    assets_mean_returns = mean(pb_struct.Assets_Returns);
    % We have two constraints: the total wealth equals 1, and the expected portfolio return should exceed the benchmark one.
    b = [1; benchmark_mean_return];
    C = pb_struct.Assets_Returns;
    A = [ones(1,n) 0; assets_mean_returns -1;];
    n = n+1;
    % The last auxiliary variable should not have any associated cost interpretation.
    C = [C sparse(l,1)]; 
    % _________________________________________________________________________________________________________________________ %
    
    % ========================================================================================================================= %
    % Complete the problem formulation based on the risk choice given by the user.
    % ------------------------------------------------------------------------------------------------------------------------- %
    if (strcmp(risk_measure.name,"CVaR"))
        % add a new auxiliary variable t (assume it is first). 
        % Add a column of -1 as the first column of C, change sign (cost) and scale by l*alpha (definition).
        n = n+1;   
        Q = sparse(n,n);
        C = (-1/(l*risk_measure.alpha)).*[ones(l,1) C];
        A = [sparse(2,1) A];
        d = zeros(l,1);
        c_2 = zeros(l,1);
        % Only the auxiliary variable appears outside the max{X,0} in the objective.
        c_1 = [1; zeros(n-1,1)];
        lb = [-Inf; risk_measure.short_cap.*ones(n-2,1); 0];
        % Apply upper bounds only for variables that correspond to an instrument.
        ub = [Inf; (risk_measure.stock_cap).*ones(n-2,1); Inf];
        % The weight matrix of the ell-1 regularization term of the objective (only for instruments).
        D = [0; risk_measure.tau.*ones(n-2,1); 0];
    elseif (strcmp(risk_measure.name,"MAsD"))
        % Change sign (cost) and scale by l (definition).
        Q = sparse(n,n);
        C(:,1:n-1) = -C(:,1:n-1) + ones(l,1)*assets_mean_returns;
        C = (1/l).*(C);
        d = zeros(l,1);
        c_2 = zeros(l,1);
        % Only the auxiliary variable appears outside the max{X,0} in the objective.
        c_1 = zeros(n,1);
        lb = [risk_measure.short_cap.*ones(n-1,1); 0];
        % Apply upper bounds only for variables that correspond to an instrument.
        ub = [(risk_measure.stock_cap).*ones(n-1,1); Inf];
        % The weight matrix of the ell-1 regularization term of the objective (only for instruments).
        D = [risk_measure.tau.*ones(n-1,1); 0];   
    end
    % _________________________________________________________________________________________________________________________ %
    
    tStart = tic;
    [x,w,v,z,opt,iter] = ADMM_warmstart(Q,D,A,A',C,C',b,d,c_1,c_2,lb,ub,tol,max_ADMM_iter);
    tEnd = toc(tStart);

    solution_statistics.total_ADMM_iters = iter;
    solution_statistics.total_time = tEnd;  
    solution_statistics.objective_value = c_1'*x + c_2'*w + (1/2)*(x'*(Q*x)) + sum(max(w,zeros(l,1))) + norm(D.*x,1);
    solution_statistics.solution_struct.x = x;
    solution_statistics.solution_struct.w = w;
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

