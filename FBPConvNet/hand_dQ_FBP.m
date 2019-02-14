function [dr, fxr] = hand_dQ_FBP(net, x0, r, lambda, theta)
    % Compute the graident of 
    % Q(r) = ||f(A(x0+r)) - f(A(x0))||_{2}^{2} - 0.5*lambda*||r||_{2}^{r}
    % and the function value f(A(x0+r)), where f is a neural network, A is 
    % the radon sampling operator with angles theta.
    %
    % INPUT
    % net - Network structure
    % x0  - Original image
    % r   - Perturbation in x0
    % lambda - Scalar
    % theta  - Sampling angels in the range [0, 180)
    %
    % OUTPUT
    % df  - Gradient w.r.t. r
    % fxr - NN reconstruction of x0+r sampled at theta
    %

    if (nargin < 3)
        r = single(0);
    end

    x1 = iradon(radon(x0+r, theta), theta, 'linear', 'Ram-Lak', 1, 512);
    
    res1 = vl_simplenn_fbpconvnet(net,gpuArray((single(x1))), single(1));
    dr = gather(iradon(radon(res1(1).dzdx, theta), theta, 'linear', 'Ram-Lak', 1, 512) - lambda*r) ;
    fxr = gather(res1(end-1).x)+ x1;

end

