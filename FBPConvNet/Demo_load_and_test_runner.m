% This script plots the cropped images of the med 50 network with and 
% without the perturbation

on_stability_config; % Load all network paths 

% 50 views
load(med_50_weights_path);

N = 512;
nbr_lines = 50;
theta = linspace(0,180*(1-1/nbr_lines), nbr_lines);

funcs.dL = @(net, x0, r) hand_dQ_FBP(net, x0, r, runner.lambda, theta);
funcs.f  = @(net, x0)    hand_f_FBP_samp2(net, x0, theta);
funcs.theta = theta;

net = vl_simplenn_move(net, 'cpu');
gpuDevice(1);
net = vl_simplenn_move(net, 'gpu');

dest = 'plots_adversarial_med_50';
if (exist(dest) ~= 7) 
    mkdir(dest);
end

run_id = 2;

% Load runner
load(fullfile(runner_path, sprintf('data/runner_%d.mat', run_id)));
x0 = runner.x0{1};
r  = runner.r{1};

fx  = funcs.f(net, x0);
fxr = funcs.f(net, x0+r);

fname1 = fullfile(dest, sprintf('r_%d_original.png', run_id));
fname2 = fullfile(dest, sprintf('r_%d_org_pert.png', run_id));
fname3 = fullfile(dest, sprintf('r_%d_rec_org.png', run_id));
fname4 = fullfile(dest, sprintf('r_%d_rec_pert.png', run_id));

x1  = scale_to_01(double(x0));
x1r = scale_to_01(double(x0+r));
fx  = scale_to_01(double(fx));
fxr = scale_to_01(double(fxr));

imwrite(im2uint8(x1), gray(256), fname1);
imwrite(im2uint8(x1r), gray(256), fname2);
imwrite(im2uint8(fx), gray(256), fname3);
imwrite(im2uint8(fxr), gray(256), fname4);

% Crop the images

fname1 = fullfile(dest, sprintf('r_%d_original_crop.png', run_id));
fname2 = fullfile(dest, sprintf('r_%d_org_pert_crop.png', run_id));
fname3 = fullfile(dest, sprintf('r_%d_rec_org_crop.png', run_id));
fname4 = fullfile(dest, sprintf('r_%d_rec_pert_crop.png', run_id));

N = 512;
s = 360;

lh = (N-s)/2;
lv = (N-s);

x1  = x1(lv+1:end, lh+1:s+lh);
x1r = x1r(lv+1:end, lh+1:s+lh);
fx  = fx(lv+1:end, lh+1:s+lh);
fxr = fxr(lv+1:end, lh+1:s+lh);

imwrite(im2uint8(x1), gray(256), fname1);
imwrite(im2uint8(x1r), gray(256), fname2);
imwrite(im2uint8(fx), gray(256), fname3);
imwrite(im2uint8(fxr), gray(256), fname4);




