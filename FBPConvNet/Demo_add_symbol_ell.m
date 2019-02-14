
on_stability_config; % Load all network paths 

load(ell_50_weights_path);

net = vl_simplenn_move(net, 'cpu');
gpuDevice(1); % Clear GPU
net = vl_simplenn_move(net, 'gpu');

N = 512;
fsize = 11;

dest  = 'plots_symbol_ell';
if (exist(dest) ~= 7) 
    mkdir(dest);
end

fname = fullfile(ell_symbol_path, sprintf('ell_symb_fsize_%02d.mat', fsize));
load(fname); % Y

Y = 1.1*Y;
irad = @(I, ang) iradon(I, ang, 'linear', 'Ram-Lak', 1, N);

f  = @(net, x0)  hand_f_FBP (net, x0);

nbr_lines = 1000;
theta = linspace(0, 180*(1-1/nbr_lines), nbr_lines);

subs = 1:20:nbr_lines; 

theta1 = theta(subs);

rec = irad(radon(Y, theta1), theta1);

fx = f(net, rec);

fname = fullfile(dest, ...
        sprintf('FBPConvNet_can_u_see_it_fsize_%02d_views_50.png', fsize));
imwrite(im2uint8(scale_to_01(fx)), gray(256), fname);



