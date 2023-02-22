% Include Code and Data files.
curr_path = pwd;
addpath(genpath(curr_path)); 
fid = 1;   

fprintf('Should the default parameters be used?\n');
default = input('Type 1 for default parameters, or anything else to manually include them.\n');
if (default == 1)
    tol = 1e-5;                                             % Tolerance used.
    max_PMM_iter = 200;                                     % Maximum number of PMM iterations.
    printlevel = 2;                                         % Printing choice (see SSN_PMM documentation).
    problem_set = "Binary_Classification";                  % Choices: "Portfolio_Optimization", "Quantile_Regression", "Binary_Classification"
    warm_start = "matrix-free";                             % Warm-start technique: "standard" for ADMM, 
                                                            % or "matrix-free" for Chambolle-Pock.
else
    fprintf('Choose a value for the allowed error tolerance.\n');
    while(true)
        tol = input('Type a double value in the form 1e-k, where k must be in [1,12].\n');
        if (isinf(tol) || isnan(tol) || ~isa(tol,'double') || tol > 1e0 || tol < 1e-12)
            fprintf('Incorrect input argument.\n');
        else
            break;
        end
    end
    fprintf('Choose the maximum number of PMM iterations.\n');
    while(true)
        max_PMM_iter = input('Type an integer value in k between [50,300].\n');
        if (isinf(max_PMM_iter) || isnan(max_PMM_iter) || floor(max_PMM_iter)~= max_PMM_iter || ...
            max_PMM_iter > 300 || max_PMM_iter < 50)
            fprintf('Incorrect input argument.\n');
        else
            break;
        end
    end
    fprintf('Choose the printlevel.\n');
    fprintf('                         0: no printing\n');
    fprintf('                         1: print PMM iterations and parameters\n');
    fprintf('                         2: also print SSN iterations\n');
    while(true)
        printlevel = input('Type an integer value in k between [0,2].\n');
        if (isinf(printlevel) || isnan(printlevel) || ...
            floor(printlevel)~= printlevel || printlevel > 4 || printlevel < 1)
            fprintf('Incorrect input argument.\n');
        else
            break;
        end
    end
end


if (problem_set == "Portfolio_Optimization")
    % Problem options: 1. -> "DowJones",   2. -> "FF49Industries", 3. -> "FTSE100",
    %                  4. -> "NASDAQ100",  5. -> "NASDAQComp",     6. -> "SP500".
    pb_name = "NASDAQComp";
    risk_measure = struct();
    risk_measure.name = "MAsD";
    risk_measure.alpha = 0.05;
    risk_measure.stock_cap = 0.6;
    risk_measure.short_cap = -1;
    risk_measure.tau = 5e-2;
    [solution_statistics_PMM]  = Portfolio_optimization_problems(pb_name,risk_measure,tol,max_PMM_iter,printlevel,fid,warm_start);
elseif (problem_set == "Quantile_Regression")
    % Problem options: 
    %    1. -> 'abalone_scale.txt',   2. -> 'cadata.txt',             3. -> 'cpusmall_scale.txt',
    %    4. -> 'E2006.train',         5. -> 'space_ga_scale.txt'.
    parameters = struct();
    pb_name = 'cadata.txt';
    parameters.alpha = 0.8;
    parameters.tau = 0.8;
    parameters.lambda = 1e-3;    
elseif (problem_set == "Binary_Classification")
    % Problem options: 
    %    1. -> 'real-sim',   2. -> 'rcv1_train.binary',             3. -> 'news20.binary',
    %    4. -> 'gisette_scale',         5. -> 'epsilon_normalized', 6. -> 'breast-cancer_scale.txt',  7. -> 'a9a.txt'.
    parameters = struct();
    pb_name = 'news20.binary';
    parameters.regularizer = "elastic"; % "elastic" -> combine regularizers, "l1" -> l1 regularizer, "l2" -> l2 regularizer.
    if (parameters.regularizer == "elastic")
        parameters.tau_1 = 0.2;
        parameters.tau_2 = 0.8;
    end
    parameters.lambda = 1e-2;
    [solution_statistics_PMM] = Binary_classification_problems(pb_name,parameters,tol,max_PMM_iter,printlevel,fid,warm_start);
end

