function run_id = save_runner(net, funcs, runner, opts, dest, network_name, nbr_angles)
    if (nargin < 6)
        network_name = '';
    end
    if (nargin < 7)
        nbr_angles = -1;
    end
    %dest = '/local/scratch/public/va304/FBPConvNet';

    fname_desc = fullfile(dest, 'data/runner_desciption.txt');
    fID = fopen(fname_desc, 'a');
    run_id = read_counter( fullfile(dest, 'COUNT.txt') );
    
    fprintf(fID, '-----------------------------------------------------------\n');
    fprintf(fID, '%s, ID: %d, %s, nbr_angles: %d\n', datestr(datetime('now')), run_id, network_name, nbr_angles);
    n = numel(runner.x0);
    
    fprintf(fID, 'max_itr: %d, ', runner.max_itr);
    fprintf(fID, 'opt: %s, ws: %s, ws_fact: %g, la: %g, l_rate: %g, mom: %g\n', ...
                runner.optimizer, runner.warm_start, runner.warm_start_factor, ...
                runner.lambda, opts.learning_rate, opts.momentum);

    for k = 1:n

        fxr = funcs.f(net,runner.x0{k});
        fx = funcs.f(net,runner.x0{k} + runner.r{k});

        norm_fx_fxr = norm(fx-fxr, 'fro');
        norm_r = norm(runner.r{k}, 'fro');
        norm_x = norm(runner.x0{k}, 'fro');

        fprintf(fID, '    im %2d, |f(x)-f(x+r)|: %g, |r|: %g, |f(x)-f(x+r)|/|r|: %g, |r|/|x0|: %g\n', ...
                k, norm_fx_fxr, norm_r, norm_fx_fxr/norm_r, norm_r/norm_x);
    end
    
    fname = sprintf('data/runner_%d.mat', run_id);
    save(fullfile(dest,fname), 'runner')
    
    fclose(fID);
end



