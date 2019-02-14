# This file requiere that you have ran 'Demo_psnr_all_nets.m' to create the file 
# 'add_more_samples/ll_psnr.mat'

# This script plots the psnr-values for the various networks.


import numpy as np;
import matplotlib.pyplot as plt;
from scipy.io import loadmat;
from os.path import join

dest = 'add_more_samples';
data = loadmat(join(dest, 'all_psnr.mat'));


mean_psnr_ell50 = data['ell50_psnr_mean'];
mean_psnr_med50 = data['med50_psnr_mean'];
#mean_psnr_med143 = data['med143_psnr_mean'];

views = data['nbr_views'];

lwidth = 2;
fsize  = 20;

p_max = max(max(mean_psnr_ell50), max(mean_psnr_med50)); #, max(mean_psnr_med143));
p_min = min(min(mean_psnr_ell50), min(mean_psnr_med50)); #, min(mean_psnr_med143));

fig = plt.figure();
plt.plot([50,50],      [0.92*p_min, 1.04*p_max], 'r--', linewidth=lwidth);
#plt.plot([143,143],    [1.02*p_min, 0.92*p_max], 'r--', linewidth=lwidth);
plt.plot(views, mean_psnr_ell50,  'b*-', linewidth=lwidth, ms=12, label='Ell 50');
plt.plot(views, mean_psnr_med50,  'm*-', linewidth=lwidth, ms=12, label='Med 50');
#plt.plot(views, mean_psnr_med143, 'y*-', linewidth=lwidth, ms=12, label='med143');


plt.legend(fontsize=fsize, loc=(0.65,0.6));
plt.axis([0, views[-1], 0.92*p_min, 1.04*p_max]);

plt.xlabel('Number of angles', fontsize=fsize);
plt.ylabel('Avrage PSNR', fontsize=fsize);
#plt.title('FBPConvNet', fontsize=fsize);
fig.savefig(join(dest, 'FBPConvNet_add_more_samples.png'),
            bbox_inches='tight');


