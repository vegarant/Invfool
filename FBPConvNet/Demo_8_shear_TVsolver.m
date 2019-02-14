on_stability_config;

dest = '/mn/sarpanitu/ansatte-u4/vegarant/adversarial2/FBPConvNet';
plt_folder = 'plots_adversarial_med_50';

runner_id = 2;

load(fullfile(path_radon_matrices, 'radonMatrix2N512_ang50.mat'));
load(fullfile(runner_path, 'data', sprintf('runner_%d.mat', runner_id)));
fprintf('runner_id: %d\n', runner_id);

im_nbr = 1;
x0 = double(runner.x0{im_nbr});
r  = double(runner.r{im_nbr});

N = 512;
m = [N,N];

ma = max(x0(:));
mi = min(x0(:));

f = 0.05*(x0-mi)/(ma -mi);
r = 0.05*(r-mi)/(ma -mi);

y           = A*f(:);    
yr          = A*( f(:) + r(:) );    

%% set up operator
B.times     = @(x) A*x;
B.adj       = @(x) A'*x;
B.mask      = NaN;

pm.sparse_param = [0, 0, 1, 1];
pm.sparse_trans = 'shearlets';
D = getShearletOperator([N,N], pm.sparse_param);
% set parameters
pm.alpha     = 0;
pm.beta      = 5e1;
pm.mu        = [0, 5e2];
pm.lambda    = 3e-4;
pm.maxIter   = 100;
pm.epsilon   = 1e-8;
pm.adaptive  = 'NewIRL1';
pm.normalize = false;
pm.solver    = 'TVsolver';

doPlot = false;
doReport = true;

% solve
out  = TVsolver(y, [N,N], B, D, pm.alpha, pm.beta, pm.mu(1), pm.mu(2), ...
                'lambda', pm.lambda, ...
                'adaptive', pm.adaptive, ...
                'f', f, ...
                'doPlot', doPlot, ...
                'doReport', doReport, ...
                'maxIter', pm.maxIter, ...
                'epsilon', pm.epsilon);

outr = TVsolver(yr, [N,N], B, D, pm.alpha, pm.beta, pm.mu(1), pm.mu(2), ...
                 'lambda', pm.lambda, ...
                 'adaptive', pm.adaptive, ...
                 'f', f+r, ...
                 'doPlot', doPlot, ...
                 'doReport', doReport, ...
                 'maxIter', pm.maxIter, ...
                 'epsilon', pm.epsilon);


fx = real(out.rec);
fxr = real(outr.rec);


fname1 = fullfile(dest, plt_folder, ...
                  sprintf('r_%d_rec_shear.png', runner_id));
fname2 = fullfile(dest, plt_folder, ...
                  sprintf('r_%d_rec_shear_pert.png', runner_id));
imwrite(im2uint8(scale_to_01(fx)), gray(256), fname1);
imwrite(im2uint8(scale_to_01(fxr)), gray(256), fname2);

N = 512;
s = 360;

lh = (N-s)/2;
lv = (N-s);

fx  = fx(lv+1:end, lh+1:s+lh);
fxr = fxr(lv+1:end, lh+1:s+lh);

fname1 = fullfile(dest, plt_folder, ...
                  sprintf('r_%d_rec_shear_crop.png', runner_id));
fname2 = fullfile(dest, plt_folder, ...
                  sprintf('r_%d_rec_shear_pert_crop.png', runner_id));
imwrite(im2uint8(scale_to_01(fx)), gray(256), fname1);
imwrite(im2uint8(scale_to_01(fxr)), gray(256), fname2);











