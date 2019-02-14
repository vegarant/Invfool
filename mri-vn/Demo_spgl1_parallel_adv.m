plot_dir = 'plots_adversarial';
samp_type = 'unif'; 
samp_frac = 0.15; 

runner_id = 4;
src = fullfile(plot_dir, sprintf('%s_%g', samp_type, samp_frac));
load(fullfile(src, 'data', sprintf('runner_%d_coil_sens.mat', runner_id)));
load(fullfile(src, 'data', sprintf('runner_%d_mask.mat', runner_id)));
load(fullfile(src, 'data', sprintf('runner_%d_r.mat', runner_id)));
load(fullfile(src, 'data', sprintf('runner_%d_x0.mat', runner_id)));
coil_sens = permute(coil_sens, [2,3,1]);

x0        = double(x0);
r         = double(r);
mask      = double(mask);
coil_sens = double(coil_sens);

m = size(x0);
m = [m(1), m(2), 15]; % [height, width, channels];
mask = repmat(mask,1,1,m(3));

%A = getFourierOperator2(m, mask);
nres = 3;
vm = 2;

U = getParallelFourierWaveOperator(m, mask, coil_sens, nres, vm);
A = getParallelFourierOperator(m, mask, coil_sens);

y = A.times(x0);
yr = A.times(x0+ r);

opA = @(x, mode) simple_spgl1_op(x, mode, U);
noise_level = 0.01;

%  minimize ||x||_1  s.t.  ||Ax - b||_2 <= noise_level
opts_spgl1 = spgSetParms('verbosity', 1);
y_out = spg_bpdn(opA, y, noise_level, opts_spgl1);
yr_out = spg_bpdn(opA, yr, noise_level, opts_spgl1);

S = U.S;
wname = U.wname;
im_rec_x  = waverec2( y_out, S, wname);
im_rec_xr  = waverec2(yr_out, S, wname);

dest = fullfile(src,'wave');
if (exist(dest) ~= 7)
    mkdir(dest);
end

fname_x = fullfile(dest,...
                sprintf('rec_rID_%d_unif_0.15_noise_%g_x.png', ...
                runner_id, noise_level));
fname_xr = fullfile(dest,...
                    sprintf('rec_rID_%d_unif_0.15_noise_%g_xr.png', ...
                    runner_id, noise_level));

fx  = scale_to_01(abs(im_rec_x));
fxr = scale_to_01(abs(im_rec_xr));
imwrite(uint8(255*fx),  fname_x);
imwrite(uint8(255*fxr), fname_xr);
