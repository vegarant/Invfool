


src = '/mn/sarpanitu/ansatte-u4/vegarant/storage/FBPConvNet/symbols/plots_ell'; 
dest = 'plots_8_symb_ell';

nbr_lines = 50;
theta = linspace(0,180*(1-1/nbr_lines), nbr_lines);
fsize = 11;

fname = fullfile( src, sprintf('ell_symb_fsize_%02d.mat', fsize) );
load(fname); % Y                                                                

load('~/storage/radon_matrices/radonMatrix2N512_ang50.mat');
theta = linspace(0,180*(1-1/nbr_lines), nbr_lines);                            

N = 512;                                                                        
m = [N,N];

ma = max(Y(:));
mi = min(Y(:));
f = double(0.05*(Y-mi)/(ma -mi));

y = A*f(:);    

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

out  = TVsolver(y, [N,N], B, D, pm.alpha, pm.beta, pm.mu(1), pm.mu(2), ...
                'lambda', pm.lambda, ...
                'adaptive', pm.adaptive, ...
                'f', f, ...
                'doPlot', doPlot, ...
                'doReport', doReport, ...
                'maxIter', pm.maxIter, ...
                'epsilon', pm.epsilon);

rec = out.rec;

fname = fullfile(dest, ...                                                      
        sprintf('FBPConvNet_can_u_see_it_fsize_%02d_views_50.png', fsize));

imwrite(im2uint8(scale_to_01(rec)), gray(256), fname);



