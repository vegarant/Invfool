import pickle
import tensorlayer as tl
import numpy as np
import os
import nibabel as nib
import tensorflow as tf
import time
import matplotlib.pyplot as plt;

from model import *
from utils import *
from config import config, log_config
from scipy.io import loadmat, savemat
from dagan_tools import *; # hand_f
from adversarial_tools import scale_to_01, compute_psnr;
from os.path import join;

   
if __name__ == "__main__":
    model_name1 = 'unet_refine'; # 'unet_refine'; 
    mask_name = 'gaussian1D';
    mask_perc = 20;
    rec_dest = 'plots_add_more_samples';
    if not os.path.isdir(rec_dest):
        os.mkdir(rec_dest);

    testing_data_path = config.TRAIN.testing_data_path;   

    with open(testing_data_path, 'rb') as f:
        batch = pickle.load(f)

    batch_size = batch.shape[0];
    print('batch_size: ', batch_size);

    mask = load_sampling_pattern(mask_name, mask_perc);

    nw, nh, nz = batch.shape[1:];

    tf_input = tf.placeholder('float32', [None, nw, nh, nz])
    network_unet_ref = load_network(model_name1, tf_input);

    sess_unet_ref = load_session(model_name1, mask_name, mask_perc, network_unet_ref);

    print('Compiling network function'); 
    def val_fn_unet_ref(x_adj):
        return sess_unet_ref.run(network_unet_ref.outputs, feed_dict={tf_input: x_adj});

    # Function handle
    f_unet_ref = lambda x, mask: hand_f(val_fn_unet_ref, x, mask);

    subsampling_fact = [5,10,20,30,40,50];
    batch_cp = batch.copy();    
    psnr_table_unet_ref = np.zeros([len(subsampling_fact), batch_size]);
    
    for k in range(len(subsampling_fact)):
        s_fact = subsampling_fact[k];
        mask   = load_sampling_pattern(mask_name, s_fact);

        reconstructions_unet_ref = f_unet_ref(batch, mask);

        for i in range(batch_size):

            im_rec_unet_ref = reconstructions_unet_ref[i,:,:,0];
            ref             = np.squeeze(batch_cp[i]); 
            psnr_unet_ref   = compute_psnr(im_rec_unet_ref, ref);
            psnr_table_unet_ref[k,i] = psnr_unet_ref; 

    psnr_mean_unet_ref = np.mean(psnr_table_unet_ref, axis=1);

    fsize  = 20;
    lwidth = 2;
    
    fig = plt.figure();
    x_axis_values = np.asarray(subsampling_fact)/100;
    plt.plot(x_axis_values, psnr_mean_unet_ref, 'b*-', linewidth=lwidth, ms=12);
    p_min = min(psnr_mean_unet_ref);
    p_max = max(psnr_mean_unet_ref);
    plt.xlabel('Subsampling rate', fontsize=fsize);
    plt.ylabel('Avrage PSNR', fontsize=fsize);
    plt.plot([mask_perc/100, mask_perc/100],[0.98*p_min, 1.02*p_max], 'r--', linewidth=lwidth);
    plt.axis([0,x_axis_values[-1], 0.98*p_min, 1.02*p_max]);
    fname = 'dagan_%d_%s_add_more_samples.png' % (mask_perc, mask_name); 
    plt.savefig(join(rec_dest, fname), bbox_inches='tight');




