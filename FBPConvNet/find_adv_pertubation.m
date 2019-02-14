% Search for an adverserial pertubation
% 
% INPUT:
% net    - MatConvNet Neural Network
% runner - Runner structure
% opts   - options for the optimization algorithm
% batch  - Batch of images in the form (height × width × channel × batch size)
%
% 
% OUTPUT
% result - Runner structure with the pertubations
%
%
% Runner needs a field containing the forward function.
%
function result = find_adv_pertubation(net, funcs, runner, opts, batch);

    N = size(batch, 1); % Images have size N × M
    M = size(batch, 2); % Images have size N × M
    n = size(batch, 4); % Number of images
    
    contains_all_fields = isfield(runner,'warm_start') & ...
                          isfield(runner,'warm_start_factor') & ...
                          isfield(runner,'p_norm') & ...
                          isfield(runner,'perp_start') & ...
                          isfield(runner,'perp_start_factor') & ...
                          isfield(runner,'max_itr') & ...
                          isfield(runner,'optimizer') & ...
                          isfield(runner,'verbose') & ...
                          isfield(funcs, 'f') & ...
                          isfield(funcs, 'dL');

    if (~contains_all_fields)
        error('Runner struct does not contain all paramters');
    end

    result = runner;

    warm_start = runner.warm_start;
    ws_factor  = single(runner.warm_start_factor);
    perp_start = runner.perp_start;
    ps_factor  = single(runner.perp_start_factor);
    p_norm     = runner.p_norm;
    max_itr    = runner.max_itr;
    optimizer  = runner.optimizer;

    if (runner.verbose)
        fprintf('------------------------------------\n');
        fprintf('Running all images with paramters:\n');
        fprintf('Optimizer:  %s\n', runner.optimizer);
        fprintf('warm_start: %s\n', warm_start);
        fprintf('ws_factor:  %g\n', ws_factor);
        fprintf('perp_start: %s\n', perp_start);
        fprintf('ps_factor:  %g\n', ps_factor);
        fprintf('p_norm:     %g\n', p_norm);
        fprintf('lambda:     %g\n', runner.lambda);
        fprintf('max_itr:    %g\n', max_itr);
        fprintf('warm_start: %s\n', warm_start);
        fprintf('ws_factor:  %g\n', ws_factor);
        fprintf('------------------------------------\n');
    end

    % Check whether r is initialized so that we can get a warm start 
    r_is_initialized = 0;
    if (isfield(result, 'r'))
        r_is_initialized = 1;
        for k = 1:n
            r_is_initialized = r_is_initialized & isnumeric(result.r{k});
        end
    end
    
    result.x0 = cell(n, 1);
    if (~r_is_initialized)
        result.r = cell(n, 1);
        rr = zeros([N,M]);
        switch lower(perp_start)
            case 'rand'
                rr = ps_factor*rand(N,M, 'single');
            case 'randn'
                rr = ps_factor*randn(N,M, 'single');
            case 'ones'
                rr = ps_factor*ones(N,M, 'single');
            case 'ellipse'
                rr = ps_factor*create_ellipse(N);
            case 'off'
                rr = zeros([N,M], 'single');
            otherwise
                error('perp start option not recognized');
        end
        for k = 1:n
            result.r{k} = rr;
        end
    end

    % If r is initialized and we are using SDG or Nesterov, we might 
    % want a varm start for v as well
    v_is_initialized = 0;
    if (isfield(result, 'v'))
        v_is_initialized = 1;
        for k = 1:n
            v_is_initialized = v_is_initialized & isnumeric(result.v{k});
        end
    end
    if (~v_is_initialized)
        result.v = cell(n, 1);
        for k = 1:n
            result.v{k} = single(0);
        end
    end

    result.norm_fx_fxr = zeros(n,1);
    result.norm_r = zeros(n,1);
    result.backlog = cell(n,1);
    
    if (isfield(runner, 'boundx'))
        boundx = runner.boundx;
    end
    if (isfield(runner, 'boundy'))
        boundy = runner.boundy;
    end
    for k = 1:n
        fprintf('-------------------- Im: %d --------------------\n', k);
        x0 = batch(:,:,:,k);

        net = vl_simplenn_move(net, 'gpu');
        %res = vl_simplenn_fbpconvnet(net, gpuArray((single(x0))), single(1));

        fx = funcs.f(net, x0);
        N_out = size(fx,1);
        M_out = size(fx,2);
       
        label = fx;
        
        switch lower(warm_start)
            case 'rand'
                label = label + ws_factor*rand(N_out, M_out);
            case 'randn'
                label = label + ws_factor*randn(N_out, M_out);
            case 'ones'
                label = label + ws_factor*ones(N_out, M_out);
            case 'zero'
                label = zeros(size(label), 'single');
            case 'off'
                % Keep label as is
            otherwise
                error('Warm start option not recognized');
        end
        net = vl_simplenn_move(net, 'cpu');
        gpuDevice([]);

        net.layers{end}.class = label;
        
        net = vl_simplenn_move(net, 'gpu');
        
        switch lower(optimizer)
            case 'SGA'
                [r1, v1, backlog] = perturb_SGA(net, funcs, x0, max_itr, opts, ...
                                                    result.r{k}, result.v{k});
            otherwise
                error('Optimizer not recognized');
        end

        result.backlog{k} = backlog;
        result.x0{k} = gather(x0);    
        result.r{k} = gather(r1);    
        if (exist('v1'))
            result.v{k} = gather(v1);
        end
        fxr = funcs.f(net, x0+r1);
        
        result.norm_fx_fxr(k) = norm(fx-fxr,'fro');
        result.norm_r(k) = norm(r1,'fro');

        n_fx_fxr = result.norm_fx_fxr(k);
        n_r      = result.norm_r(k);
        n_x      = norm(result.x0{k}, 'fro');

        if (runner.verbose)
            fprintf('im%2d: ||f(x)-f(x+r)||: %g, ||r||: %g, ||f(x)-f(x+r)||/||r|| : %g, |r|/|x|: %g\n', ...
            k, n_fx_fxr, n_r, n_fx_fxr/n_r, n_r/n_x);
        end
    end
    
    net = vl_simplenn_move(net, 'cpu');
    gpuDevice([]);
end




