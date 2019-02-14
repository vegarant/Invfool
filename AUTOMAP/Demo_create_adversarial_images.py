"""
This file creates the adversarial noise images. It must be ran after 
the file `Demo_adversarial_noise_multi.py` have generated a runner object. 
The relevant Runner ID must be specified, and in the file automap_config.py, 
the relevant paths must be updated
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


sess = tf.Session();

raw_f, raw_df = compile_network(sess, batch_size);

f  = lambda x: hand_f( raw_f, x, k_mask_idx1, k_mask_idx2);

runner_id = 52;


runner1 = load_runner(runner_id);

if (runner_id <= 35):
    print("""
runner_id:         %d
la:                %g
learning_rate:     %g
momentun:          %g
perp_start:        %s
perp_start_factor: %g
""" % (runner_id, runner1.la, runner1.learning_rate, 
       runner1.momentum, runner1.perp_start,
       runner1.perp_start_factor));
else:
    print(runner1);


mri_data = runner1.x0[0];
bd = 5;
N = 128;
for i in range(len(runner1.r)):
    rr = runner1.r[i];
    if i == 0:
        rr = np.zeros(rr.shape, dtype=rr.dtype);
    fxr = f(mri_data + rr);
    for im_nbr in [3,4,5]:    
        x = mri_data[im_nbr, :,:];
        r = rr[im_nbr, :,:];
        fxr1 = fxr[im_nbr, :,:];
        im_left  = scale_to_01(abs(x+r));
        im_right = scale_to_01(fxr1);
        im_out = np.ones([N, 2*N + bd]);
        im_out[:,:N] = im_left;
        im_out[:,N+bd:] = im_right;
        fname_out = join(plot_dest, \
                         'rec_automap_runner_%d_int_%d_idx_%d.png' % (runner_id, im_nbr, i));
        
        plt.imsave(fname_out, im_out, cmap='gray');
        
        fname_out_noisy = join(plot_dest, splits, \
                               'runner_%d_int_%d_idx_%d_noisy.png' % (runner_id, im_nbr, i));
        fname_out_noisy_rec = join(plot_dest, splits, \
                               'runner_%d_int_%d_idx_%d_noisy_rec.png' % (runner_id, im_nbr, i));
        
        plt.imsave(fname_out_noisy, im_left, cmap='gray');
        plt.imsave(fname_out_noisy_rec, im_right, cmap='gray');



sess.close();

















