on_stability_config; % Load all network paths 

% 50 views
load(med_50_weights_path);

nbr_images = 1;
N = 512;
batch = zeros([N, N, 1, nbr_images]);

%file_nbr = 480;

for i = 1:nbr_images
    file_nbr = 475 + i;
    data_name = fullfile(med_data_path, sprintf('im%d.mat', file_nbr));
    load(data_name, 'GT');
    batch(:,:,1, i) = GT;
end

N = 512;
nbr_lines = 50;
theta = linspace(0,180*(1-1/nbr_lines), nbr_lines);

runner.verbose = 1;
runner.p_norm = 2;
runner.lambda = 20; 
runner.max_itr = 100;
runner.warm_start = 'off';
runner.warm_start_factor = 0;
runner.perp_start = 'rand';
runner.perp_start_factor = 0.005;
runner.optimizer = 'SGA';

opts.momentum = 0.9;
opts.smoothing_eps = 1e-8;
opts.verbose = 1;
opts.learning_rate = 0.005;
opts.max_r_norm = 1800;
opts.max_diff_norm = 50000;

funcs.dL = @(net, x0, r) hand_dQ_FBP(net, x0, r, runner.lambda, theta);
funcs.f  = @(net, x0)    hand_f_FBP_samp2(net, x0, theta);
funcs.theta = theta;

verbose = 1;
view_image = 1;

%%vl_simplenn_display(net);

result = find_adv_pertubation(net, funcs, runner, opts, batch);

%% % Save result
net = vl_simplenn_move(net, 'cpu');
gpuDevice(1);
net = vl_simplenn_move(net, 'gpu');

dest = 'plots_adversarial_med_50';
if (exist(dest) ~= 7) 
    mkdir(dest);
end

x0 = result.x0{1};
r  = result.r{1};

runner_id = save_runner(net, funcs, result, opts, runner_path, 'med50', 50);
runner_id

fx  = funcs.f(net, x0);
fxr = funcs.f(net, x0+r);

fname1 = fullfile(dest, 'original.png');
fname2 = fullfile(dest, 'org_pert.png');
fname3 = fullfile(dest, 'rec_org.png');
fname4 = fullfile(dest, 'rec_pert.png');

imwrite(im2uint8(scale_to_01(double(x0))), gray(256), fname1);
imwrite(im2uint8(scale_to_01(double(x0+r))), gray(256), fname2);
imwrite(im2uint8(scale_to_01(double(fx))), gray(256), fname3);
imwrite(im2uint8(scale_to_01(double(fxr))), gray(256), fname4);















