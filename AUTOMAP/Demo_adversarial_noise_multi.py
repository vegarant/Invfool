"""
This script searches for a perturbation, which makes the network fail.
The result will be saved in a Runner object. Make sure you have updated the 
automap_config.py file before running this script.
"""

import tensorflow as tf;
import scipy.io;
import h5py
from os.path import join;
import os;
import os.path;
import _2fc_2cnv_1dcv_L1sparse_64x64_tanhrelu_upg as arch
import matplotlib.image as mpimg;
import numpy as np;
from automap_config import src_weights, src_mri_data;
from automap_tools import *;
from Runner import Runner;
from Automap_Runner import Automap_Runner;

k_mask_idx1, k_mask_idx2 = read_automap_k_space_mask();

data = scipy.io.loadmat(join(src_mri_data, 'HCP_mgh_1033_T2_128_w_symbol.mat'));
mri_data = data['mr_images_w_symbol'];

batch_size = mri_data.shape[0];

# Plot parameters
N = 128; # out image shape
bd = 5;  # Boundary between images
plot_dest = './plots_con';
splits = 'splits';

if not (os.path.isdir(plot_dest)):
    os.mkdir(plot_dest);
    split_dest = join(plot_dest, splits);
    if not (os.path.isdir(split_dest)):
        os.mkdir(split_dest);

# Optimization parameters
max_itr = 8; # Update list below. This value is not relevant here
max_r_norm = float('Inf');
max_diff_norm = float('Inf');
la = 0.1;
warm_start = 'off';
warm_start_factor = 0.0;
perp_start = 'rand';
perp_start_factor = 1e-5;
reference = 'true';
momentum = 0.9;
learning_rate = 0.001;
verbose=True; 

sess = tf.Session();

raw_f, raw_df = compile_network(sess, batch_size);

f  = lambda x: hand_f( raw_f, x, k_mask_idx1, k_mask_idx2);
dQ = lambda x, r, label, la: hand_dQ(raw_df, x, r, label, la, 
                                          k_mask_idx1, k_mask_idx2); 

runner = Automap_Runner(max_itr, max_r_norm, max_diff_norm, 
                         la=la, 
                         warm_start=warm_start,
                         warm_start_factor=warm_start_factor,
                         perp_start=perp_start,
                         perp_start_factor=perp_start_factor,
                         reference=reference,
                         momentum=momentum,
                         learning_rate= learning_rate,
                         verbose=verbose,
                         mask= [k_mask_idx1, k_mask_idx2]
                         );

# Update the number of iteration you would like to run
max_itr_schedule = [12, 4, 4, 4];

for i in range(len(max_itr_schedule)):
    max_itr = max_itr_schedule[i];
    runner.max_itr = max_itr;
    runner.find_adversarial_perturbation(f, dQ, mri_data);

runner_id = runner.save_runner(f);

print('Saving runner as nbr: %d' % runner_id);
runner1 = load_runner(runner_id);

mri_data = runner1.x0[0];
im_nbr = 5;
bd = 5;
N = 128;
for i in range(len(runner1.r)):
    rr = runner1.r[i];
    if i == 0:
        rr = np.zeros(rr.shape, dtype=rr.dtype);
    fxr = f(mri_data + rr);
    x = mri_data[im_nbr, :,:];
    r = rr[im_nbr, :,:];
    fxr = fxr[im_nbr, :,:];
    im_left  = scale_to_01(abs(x+r));
    im_right = scale_to_01(fxr);
    im_out = np.ones([N, 2*N + bd]);
    im_out[:,:N] = im_left;
    im_out[:,N+bd:] = im_right;
    fname_out = join(plot_dest, \
                     'rec_automap_runner_%d_r_idx_%d.png' % (runner_id, i));
    plt.imsave(fname_out, im_out, cmap='gray');
    fname_out_noisy = join(plot_dest, splits, \
                           'runner_%d_r_idx_%d_noisy.png' % (runner_id, i));
    fname_out_noisy_rec = join(plot_dest, splits, \
                           'runner_%d_r_idx_%d_noisy_rec.png' % (runner_id, i));
    plt.imsave(fname_out_noisy, im_left, cmap='gray');
    plt.imsave(fname_out_noisy_rec, im_right, cmap='gray');

sess.close();
