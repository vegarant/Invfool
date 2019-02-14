"""
This script searches for an instability using the algorithm in the paper.  
"""

import os
import sys
import argparse
import glob
import traceback
import time
from os.path import join;

import vn
import tensorflow as tf
import matplotlib.pyplot as plt;
from variational_tools import get_model_name_and_mask_name, hand_f, hand_dQ;

import numpy as np
from mridata import VnMriReconstructionData
from Variational_Runner import Variational_Runner;
import mriutils
from os.path import join;



if __name__ == '__main__':

    samp_type = 'unif'; 
    samp_frac = 0.15; # 0.15
    print('Samp type: %s, samp frac: %g' % (samp_type, samp_frac));

    model_name, mask_name = get_model_name_and_mask_name(samp_type, samp_frac);

    path_image_config = 'configs/reco.yaml';

    dest = 'plots_adversarial';
    if not os.path.isdir(dest):
        os.mkdir(dest);

    data_config = tf.contrib.icg.utils.loadYaml(path_image_config,
                                                ['data_config'])
    data_config['dataset']['mask'] = mask_name;

    epoch = 1000; 

    checkpoint_config = tf.contrib.icg.utils.loadYaml('./configs/training.yaml',
                                                      ['checkpoint_config']);

    ckpt_dir = checkpoint_config['log_dir'] + '/' + model_name + '/checkpoints/'

    sess = tf.Session();
    #with tf.Session() as sess:
    try:
        # load from checkpoint if required
        epoch = vn.utils.loadCheckpoint(sess, ckpt_dir, epoch=epoch)
    except Exception as e:
        print(traceback.print_exc())

    # extract operators and variables from the graph
    u_op  = tf.get_collection('u_op')[0]
    u_var = tf.get_collection('u_var')
    c_var = tf.get_collection('c_var')
    m_var = tf.get_collection('m_var')
    f_var = tf.get_collection('f_var')
    g_var = tf.get_collection('g_var')

    # create data object
    data = VnMriReconstructionData(data_config,
                                   u_var=u_var,
                                   f_var=f_var,
                                   c_var=c_var,
                                   m_var=m_var,
                                   g_var=g_var,
                                   load_eval_data=False,
                                   load_target=True)

    # load data
    kspace, coil_sens, x_adj, ref, mask, norm \
                    = data.get_test_data(data_config['dataset'],
                                              data_config['dataset']['patient'],
                                              data_config['dataset']['slice']);

    # compile functions
    def val_df(x_adj, kspace, coil_sens, mask, label):
        tf_label = tf.placeholder(u_op.dtype);
        loss = tf.nn.l2_loss(tf.abs(tf_label- u_op));
        grad = tf.gradients(loss, u_var[0]);
        df = sess.run(grad, feed_dict={tf_label : label,
                                         c_var[0] : coil_sens,
                                         m_var[0] : mask, 
                                         f_var[0] : kspace, 
                                         u_var[0] : x_adj,
                                         g_var[0] : ref});
        
        return df[0];

    def val_fn(x_adj, kspace, coil_sens, mask):

        u_i = sess.run(u_op, feed_dict={c_var[0] : coil_sens,
                                        m_var[0] : mask, 
                                        f_var[0] : kspace, 
                                        u_var[0] : x_adj,
                                        g_var[0] : ref});

        return u_i;

    # Function handles
    f  = lambda x : hand_f(val_fn, x, coil_sens, mask);
    dQ = lambda x, r, pred, la: hand_dQ(val_df, x, r, pred, la, 
                                        coil_sens, mask); 

    max_itr = 25; 
    max_r_norm = float('inf');

    max_diff_norm = float('inf');
    la = 1;                   
    perp_start_factor = 0.001; 

    learning_rate = 0.005;

    runner = Variational_Runner(max_itr, max_r_norm, max_diff_norm, 
                                la=la,
                                perp_start_factor=perp_start_factor, 
                                learning_rate=learning_rate,
                                mask=[mask], 
                                coil_sens=coil_sens);

    runner.find_adversarial_perturbation(f, dQ, ref);
    runner_path = os.path.join(data_config['base_dir'], 'runner');
    count = runner.save_runner(runner_path);
    print('Runner ID: %d' % count);
    #rr = runner.r[-1];

    #x  = abs(np.squeeze(ref));
    #xr = abs(np.squeeze(ref+rr));
    #fx  = abs(np.squeeze(f(ref)));
    #fxr = abs(np.squeeze(f(ref+rr))); 

    #mi = 0;
    #ma = np.amax(x);

    ## scale to [0,1]
    #x   = (x-mi)/(ma-mi);
    #xr  = (xr-mi)/(ma-mi);
    #fx  = (fx-mi)/(ma-mi);
    #fxr = (fxr-mi)/(ma-mi);

    #x[x>1] = 1;
    #xr[xr>1] = 1;
    #fx[fx>1] = 1;
    #fxr[fxr>1] = 1;

    #length, width = x.shape; 
    #bd = 5;

    #outim = np.ones([2*length + bd, 2*width + bd], dtype=x.dtype);
    #outim[:length, :width] = x;
    #outim[:length, width+bd:] = xr;
    #outim[length+bd:, :width] = fx;
    #outim[length+bd:, width+bd:] = fxr;

    #fname = 'im_max_itr_%d_la_%f_lr_%f.png' % (max_itr, la, learning_rate);
    #dest = os.path.join(dest, '%s_%g' % (samp_type, samp_frac));
    #
    #if not os.path.isdir(dest):
    #    os.mkdir(dest);
    #
    #plt.imsave(join(dest, fname), outim, cmap='gray');
