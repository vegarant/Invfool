deep_mri_config;

N = 256;

load(fullfile(mask_path, 'mask.mat'));

mask = double(mask);
mask = logical(fftshift(squeeze(mask(1,:,:))));

fsize = 10;

fname = sprintf('deep_symb_fsize_%02d.mat', fsize);
fname = fullfile(symbol_path, fname);
load(fname); % Y
Y = double(squeeze(Y(1,:,:)));

N = size(Y,1);

m = [N,N];

ma = max(Y(:));
mi = min(Y(:));

Y = (Y - mi)/(ma-mi);
% fetch Operators
A = getFourierOperator([N,N], mask);
% D = getWaveletOperator(m,2,3);
pm.sparse_param = [1, 1, 2];
pm.sparse_trans = 'shearlets';
pm.solver    = 'TGVsolver';
D = getShearletOperator([N,N], pm.sparse_param);

% fetch measurement operator
y       = A.times(Y(:));

% set parameters
pm.beta        = 1e5;
pm.alpha       = [1 1];
pm.mu          = [5e3, 1e1, 2e1];
pm.epsilon     = 1e-5;

pm.maxIter     = 500;
pm.adaptive    = 'NewIRL1';
correct     = @(x) real(x);
doTrack     = true;
doPlot      = false;
%% solve
out = TGVsolver(y, [N,N], A, D, pm.alpha, pm.beta, pm.mu, ...
                'maxIter',  pm.maxIter, ...
                'adaptive', pm.adaptive, ...
                'f',        Y, ...
                'correct',  correct, ...
                'epsilon',  pm.epsilon, ...
                'doTrack',  doTrack, ...
                'doPlot',   doPlot);


plt_fldr = 'plots_symbol';

im = abs(out.rec);

fname_out = sprintf('rec_cs_fsize_%d.png', fsize);
imwrite(im2uint8(im), fullfile(location_of_folder, plt_fldr, fname_out));










