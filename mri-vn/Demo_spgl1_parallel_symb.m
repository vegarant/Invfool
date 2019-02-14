src_data = '/local/scratch/public/va304/storage2/mri-vn/coronal_pd_fs';

esprit_nbr = 21;
%load(fullfile(src_data, sprintf('21/rawdata%d.mat', esprit_nbr)));
load(fullfile(src_data, sprintf('21/espirit%d.mat', esprit_nbr))); % reference, sensitivities
load(fullfile(src_data, 'masks/un_samp_patt_0.15.mat')); % mask

plot_dir = 'plots_symbol';
samp_type = 'unif'; 
samp_frac = 0.15; % 0.15

src = fullfile(plot_dir, sprintf('%s_%g', samp_type, samp_frac));
coil_sens = sensitivities; %permute(sensitivities, [2,3,1]);

mask      = double(mask);
coil_sens = double(coil_sens);

x0 = double(real(reference));
m = size(x0);
m = [m(1), m(2), 15]; % [height, width, channels];
mask = repmat(mask,1,1,m(3));

ma = max(x0(:));
mi = min(x0(:));

x0 = (x0 - mi)/(ma-mi);
% fetch Operators

%A = getFourierOperator2(m, mask);
nres = 3;
vm = 2;

U = getParallelFourierWaveOperator(m, mask, coil_sens, nres, vm);
A = getParallelFourierOperator(m, mask, coil_sens);

y = A.times(x0);

opA = @(x, mode) simple_spgl1_op(x, mode, U);
noise_level = 0.01;

%  minimize ||x||_1  s.t.  ||Ax - b||_2 <= noise_level
opts_spgl1 = spgSetParms('verbosity', 1);
y_out = spg_bpdn(opA, y, noise_level, opts_spgl1);

S = U.S;
wname = U.wname;
im_rec_x  = waverec2( y_out, S, wname);

dest = fullfile(src,'wave');
if (exist(dest) ~= 7)
    mkdir(dest);
end


fname_x = fullfile(dest,...
                sprintf('rec_can_u_see_it_esp%d_unif_0.15_noise_%g_x.png', ...
                esprit_nbr, noise_level));

fx  = scale_to_01(abs(im_rec_x));
imwrite(uint8(255*fx),  fname_x);
