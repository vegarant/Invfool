deep_mri_config;
plot_fldr = 'plots_adv';

N = 256;
runner_id = 24;

load(sprintf('plots_adv/runner_%d_x0.mat', runner_id));
load(sprintf('plots_adv/runner_%d_r.mat', runner_id));
load(sprintf('plots_adv/runner_%d_mask.mat', runner_id));

im_nbr = 1;
x0 = double(squeeze(x0(im_nbr, :,:)));
r  = double(squeeze(r(im_nbr, :,:)));

mask = double(mask);
mask = logical(fftshift(squeeze(mask(im_nbr,:,:))));

m = [N,N];

ma = max(x0(:));
mi = min(x0(:));

x0 = (x0 - mi)/(ma-mi);
r  = (r - mi)/(ma-mi);
% fetch Operators
A = getFourierOperator([N,N], mask);
% D = getWaveletOperator(m,2,3);
pm.sparse_param = [1, 1, 2];
pm.sparse_trans = 'shearlets';
pm.solver    = 'TGVsolver';
D = getShearletOperator([N,N], pm.sparse_param);

% fetch measurement operator
y       = A.times(x0(:));
yr       = A.times(x0(:)+r(:));

% set parameters
pm.beta        = 1e5;
pm.alpha       = [1 1];
pm.mu          = [5e3, 1e1, 2e1];
pm.epsilon     = 1e-5;

pm.maxIter     = 500;
pm.adaptive    = 'NewIRL1';
%correct     = @(x) real(x);
doTrack     = true;
doPlot      = false;



%% solve
outy = TGVsolver(y, [N,N], A, D, pm.alpha, pm.beta, pm.mu, ...
                'maxIter',  pm.maxIter, ...
                'adaptive', pm.adaptive, ...
                'f',        abs(x0), ...
                'epsilon',  pm.epsilon, ...
                'doTrack',  doTrack, ...
                'doPlot',   doPlot);

im1 = abs(outy.rec);

fname1 = fullfile(location_of_folder, plot_fldr, ...
                 sprintf('rec_cs_x_run_%d.png', runner_id));

imwrite(im2uint8(im1), fname1);

outyr = TGVsolver(yr, [N,N], A, D, pm.alpha, pm.beta, pm.mu, ...
                'maxIter',  pm.maxIter, ...
                'adaptive', pm.adaptive, ...
                'f',        abs(x0+r), ...
                'epsilon',  pm.epsilon, ...
                'doTrack',  doTrack, ...
                'doPlot',   doPlot);

fname2 = fullfile(location_of_folder, plot_fldr, ...
                 sprintf('rec_cs_xr_run_%d.png', runner_id));

im2 = abs(outyr.rec);

imwrite(im2uint8(im2), fname2);








