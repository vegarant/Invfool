function [r, v, backlog] = perturb_SGA(net, funcs, x0, epoch, opts, r0, v0) 

    if ~isstruct(opts) && ~isobject(opts)
        error('OPTS must be a structure');
    end

    if (isfield(opts, 'verbose'))
        verbose = opts.verbose;
    else 
        verbose = 1;
    end

    if (isfield(opts, 'momentum'))
        momentum = single(opts.momentum);
    else 
        momentum = single(0.9);
    end

    if (isfield(opts, 'learning_rate'))
        l_rate = single(opts.learning_rate);
    else 
        l_rate = single(0.01);
    end

    if (isfield(opts, 'adjust_lr_at_itr'))
        adjust_lr_at_itr = opts.adjust_lr_at_itr;
    else 
        adjust_lr_at_itr = -1;
    end

    if (isfield(opts, 'lr_decay_factor'))
        lr_decay_factor = opts.lr_decay_factor;
    else 
        lr_decay_factor = 1;
    end

    if (isfield(opts, 'max_r_norm'))
        max_r_norm = opts.max_r_norm;
    else 
        max_r_norm = Inf;
    end

    if (isfield(opts, 'max_diff_norm'))
        max_diff_norm = opts.max_diff_norm;
    else 
        max_diff_norm = Inf;
    end

    if (verbose)
        fprintf('------------------------------------\n');
        fprintf('Running SDG with paramters:\n');
        fprintf('Momentum:      %g\n', momentum);
        fprintf('Learning rate: %g\n', l_rate);
        fprintf('max_r_norm:    %g\n', max_r_norm);
        fprintf('max_diff_norm: %g\n', max_diff_norm);
        fprintf('------------------------------------\n');
    end

    fx = funcs.f(net, x0);

    if (nargin > 5)
        r = single(r0);
    else 
        r = funcs.dL(net, x0, single(0));
    end
    v = r;
    if (nargin > 6)
        v = single(v0);
    end


    i = 1;
    norm_fx_fxr = 0;
    norm_r = 0;
    backlog = '';
    norm_x0 = norm(x0, 'fro');
    while (i <= epoch & norm_fx_fxr < max_diff_norm & norm_r < max_r_norm) 

        [dr, fxr] = funcs.dL(net, x0, r);
        v = momentum*v + l_rate*dr;
        r = r + v;
        
        norm_fx_fxr = norm(fx-fxr, 'fro');
        norm_r = norm(r, 'fro');
        
        next_str = ... 
        sprintf('%2d: |f(x)-f(x+r)|: %8g, |r|: %8g, |f(x)-f(x+r)|/|r| : %8g, |r|/|x|: %8g\n', ...
                i, norm_fx_fxr, norm_r, norm_fx_fxr/norm_r, norm_r/norm_x0);
        
        backlog = [backlog, next_str];

        if (verbose)
            fprintf('%s', next_str);
        end
        i = i + 1;
    end
end

