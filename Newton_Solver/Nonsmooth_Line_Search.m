function alpha = Nonsmooth_Line_Search(NS,x,w,v,dx,dw,dv,mu,delta)
% ==================================================================================================================== %
% Nonsmooth Line Search
% -------------------------------------------------------------------------------------------------------------------- %
% alpha = Nonsmooth_Line_Search(NS,x,w,v,dx,dw,dv,mu,delta)
%                                                takes as an input the Newton structure (NS), containing all 
%                                                relevant information needed to build the natural residual, point 
%                                                (x,w,v) which is the point upon the residual is evaluated 
%                                                (as well as its directional derivative), point (dx,dw,dv) 
%                                                which is the computed Newton direction,
%                                                mu, which is the fraction of decrease in the 
%                                                merit function, predicted by linear extrapolation, that we 
%                                                are willing to accept, and finally, delta, which is the 
%                                                maximum possible step-length.
%                                                It returns the final step-length which satisfies the line-search
%                                                requirements.
%                                                           
% Author: Spyridon Pougkakiotis.
% ____________________________________________________________________________________________________________________ %

    % ================================================================================================================ %
    % Evaluate the natural residual at (x,w,v) and store the value of the merit function.
    % ---------------------------------------------------------------------------------------------------------------- %
    M_hat = Compute_FOC_vector(NS,x,w,v);
    Theta = norm(M_hat,2)^2;
    % ________________________________________________________________________________________________________________ %
    
    % ================================================================================================================ %
    % Let alpha = delta, and evaluate Theta(x+alpha*dx, w+alpha*dw, v+alpha*dv).
    % ---------------------------------------------------------------------------------------------------------------- %
    alpha = delta;
    x_new = x + alpha.*dx;
    w_new = w + alpha.*dw;
    v_new = v + alpha.*dv;
    M_hat_new = Compute_FOC_vector(NS,x_new,w_new,v_new);
    % ________________________________________________________________________________________________________________ %
    
    % ================================================================================================================ %
    % Iterate until you find a step-length satisfying the Armijo-Goldstein condition.
    % ---------------------------------------------------------------------------------------------------------------- %
    counter = 1;
    while (norm(M_hat_new,2)^2 > (1-2*mu*alpha)*Theta)
        counter = counter + 30;
        alpha = delta^counter;
        x_new = x + alpha.*dx;
        w_new = w + alpha.*dw;
        v_new = v + alpha.*dv;
        M_hat_new = Compute_FOC_vector(NS,x_new,w_new,v_new);
        if (alpha < 1e-3)
            break;
        end
    end
    % ________________________________________________________________________________________________________________ %
end
% ******************************************************************************************************************** %
% END OF FILE.
% ******************************************************************************************************************** %

