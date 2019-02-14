% Computes the psnr of 25 images for the med50 and ell50.
% Note that not all 25 images in for the med 50 network have been 
% made publicly available. 


on_stability_config; % Load all network paths 

load(ell_data_sinogram_full_path); % samp_pattern_full, sinogram_full

N = 512;
nbr_images = 25;
subs_sampling = 30:-1:2;
n = numel(subs_sampling); % Number of different sampling patterns 
theta_cell = cell(n,1);
nbr_views  = zeros(n,1);
for i = 1:n
    subs = subs_sampling(i);
    theta_cell{i} = samp_pattern_full(1:subs:end, 1);
    nbr_views(i) = numel(theta_cell{i});
end


% Define funcion handles
irad = @(x,th) iradon(x, th, 'linear', 'Ram-Lak', 1, N);
f = @(net, x) hand_f_FBP(net, x);

% Do calculations for Ellipsis images
load(ell_50_weights_path);
net = vl_simplenn_move(net, 'cpu');
gpuDevice(1);
net = vl_simplenn_move(net, 'gpu');


theta_full = samp_pattern_full(:,1);
ell50_snr = zeros([n, nbr_images]);
ell50_psnr = zeros([n, nbr_images]);
for im_nbr = 1:nbr_images
    gt = irad(sinogram_full(:,:,im_nbr), theta_full);
    gt_min = min(gt(:));
    gt = gt - gt_min; % Ensure that ground truth lies in the interval 
                      % [0, new max(gt(:))];
    for i = 1:n
        subs = subs_sampling(i);
        theta = theta_cell{i};
        fx = f(net, irad(sinogram_full(:,1:subs:end, im_nbr), theta));
        fx = double(fx);
        min_fx = min(fx(:));
        fx = fx - gt_min;
        [psnr1, snr1] = psnr(fx,gt, max(gt(:)));
        ell50_snr(i,  im_nbr)  = snr1;
        ell50_psnr(i, im_nbr) = psnr1;
        fprintf('%d: psnr: %g, snr: %g, mi_gt: %g gt: [%g, %g], f: [%g, %g]\n',...
            i, psnr1, snr1, gt_min, min(gt(:)), max(gt(:)), min(fx(:)), max(fx(:)) );
        %progressbar((im_nbr-1)*n+i, n*nbr_images);
    end
end

ell50_snr_mean  = mean(ell50_snr, 2);
ell50_psnr_mean = mean(ell50_psnr, 2);

% Load medical data
batch = zeros([N,N, nbr_images]);

for i = 1:nbr_images
    file_nbr = 475;
    data_name = fullfile(med_data_path, sprintf('im%d.mat', file_nbr+i));
    load(data_name, 'GT');
    batch(:,:,i) = GT;
end


% Delete old network and clear gpu
clear net;
gpuDevice(1); 

% 50 views
load(med_50_weights_path);
net = vl_simplenn_move(net, 'gpu');

fprintf('\n\n\nMEDICAL IMAGES \n\n\n');

med50_snr = zeros([n, nbr_images]);
med50_psnr = zeros([n, nbr_images]);
for im_nbr = 1:nbr_images
    gt = batch(:,:, im_nbr);
    for i = 1:n
        theta = theta_cell{i};
        fx = f(net, irad(radon(gt, theta), theta));
        fx = double(fx);
        [psnr1, snr1] = psnr(fx, gt, max(gt(:)));
        med50_snr(i,  im_nbr) = snr1;
        med50_psnr(i, im_nbr) = psnr1;
        fprintf('%d: psnr: %g, snr: %g, mi_gt: %g gt: [%g, %g], f: [%g, %g]\n',...
            i, psnr1, snr1, gt_min, min(gt(:)) - gt_min, max(gt(:)) - gt_min, min(fx(:)), max(fx(:)) );
        progressbar((im_nbr-1)*n+i, n*nbr_images);
    end
end

med50_snr_mean  = mean(med50_snr, 2);
med50_psnr_mean = mean(med50_psnr, 2);

% Create destination for the plots
dest = 'add_more_samples';
if (exist(dest) ~= 7) 
    mkdir(dest);
end

save(fullfile(dest,'all_psnr.mat'), 'ell50_psnr_mean', 'med50_psnr_mean', 'nbr_views');

