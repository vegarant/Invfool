"""
This file will load the runner object we used in the paper and create relevant 
images. It will also write the data from this runner object into a matlab 
friendly format, so that we can reconstruct it using the 
Demo_spgl1_parallel_symb.m script.
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
from variational_tools import get_model_name_and_mask_name, hand_f, load_runner;
import scipy;

import numpy as np
from mridata import VnMriReconstructionData
from Variational_Runner import Variational_Runner;
import mriutils
from os.path import join;

if __name__ == "__main__":

    runner_id = 4;

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
    
    # Load runner object with data
    runner_path = join(data_config['base_dir'], 'runner');
    runner = load_runner(runner_id, runner_path);


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

    def val_fn(x_adj, kspace, coil_sens, mask):

        u_i = sess.run(u_op, feed_dict={c_var[0] : coil_sens,
                                        m_var[0] : mask, 
                                        f_var[0] : kspace, 
                                        u_var[0] : x_adj,
                                        g_var[0] : runner.x0[0]});

        return u_i;

    # Function handles
    f  = lambda x : hand_f(val_fn, x, coil_sens, mask);

    runner_path = join(data_config['base_dir'], 'runner');
    runner = load_runner(runner_id, runner_path);
    print(
"""
Runner: %d

lambda:        %g
momentum:      %g
learning_rate: %g
ps:            %s
ps_factor:     %g
ws:            %s
ws_factor:     %g
max_itr:       %g
""" % (runner_id, runner.la, runner.momentum, 
       runner.learning_rate, 
       runner.perp_start, runner.perp_start_factor,
       runner.warm_start, runner.warm_start_factor,
       runner.max_itr));

    
    coil_sens = runner.coil_sens;
    mask = runner.mask[0];
    rr = runner.r[-1];
    print('rr.shape: ', rr.shape)
    N = rr.shape[1];
    M = rr.shape[2];
    rr[:, 0:int(0.35*N),:] = 0;
    rr[:, int(0.75*N):,:] = 0;
    rr[:,:,:int(0.35*M)] = 0;
    rr[:,:,int(0.7*M):] = 0;
    x0 = runner.x0[0];
    
    x  = abs(np.squeeze(x0));
    xr = abs(np.squeeze(x0+rr));
    fx  = abs(np.squeeze(f(x0)));
    fxr = abs(np.squeeze(f(x0+rr))); 

    mi = 0;
    ma = np.amax(x);

    # scale to [0,1]
    x   = (x-mi)/(ma-mi);
    xr  = (xr-mi)/(ma-mi);
    fx  = (fx-mi)/(ma-mi);
    fxr = (fxr-mi)/(ma-mi);

    x[x>1] = 1;
    xr[xr>1] = 1;
    fx[fx>1] = 1;
    fxr[fxr>1] = 1;

    length, width = x.shape; 
    bd = 5;

    outim = np.ones([2*length + bd, 2*width + bd], dtype=x.dtype);
    outim[:length, :width] = x;
    outim[:length, width+bd:] = xr;
    outim[length+bd:, :width] = fx;
    outim[length+bd:, width+bd:] = fxr;

    fname = 'runner_%d.png' % (runner_id);
    dest = os.path.join(dest, '%s_%g' % (samp_type, samp_frac));
    
    if not os.path.isdir(dest):
        os.mkdir(dest);
    data_dest = os.path.join(dest, 'data');
    if not os.path.isdir(data_dest):
        os.mkdir(data_dest);
   

    scipy.io.savemat(join(data_dest, 'runner_%d_x0.mat' % runner_id), mdict={'x0': np.squeeze(x0)});
    scipy.io.savemat(join(data_dest, 'runner_%d_r.mat' % runner_id),  mdict={'r':  np.squeeze(rr) });
    scipy.io.savemat(join(data_dest, 'runner_%d_coil_sens.mat' % runner_id),  mdict={'coil_sens':  np.squeeze(coil_sens) });
    scipy.io.savemat(join(data_dest, 'runner_%d_mask.mat' % runner_id),  mdict={'mask':  np.squeeze(mask) });

    plt.imsave(join(dest, fname), outim, cmap='gray');
        
    split_dest = os.path.join(dest, 'splits');
    if not os.path.isdir(split_dest):
        os.mkdir(split_dest);


    N = fx.shape[0];
    fx_crop = fx[int(N/4):int(3*N/4),:];
    fxr_crop = fxr[int(N/4):int(3*N/4),:];

    plt.imsave(join(split_dest, 'adv_%d_x.png'   % (runner_id)), x, cmap='gray');
    plt.imsave(join(split_dest, 'adv_%d_xr.png'  % (runner_id)), xr, cmap='gray');
    plt.imsave(join(split_dest, 'adv_%d_fx.png'  % (runner_id)), fx, cmap='gray');
    plt.imsave(join(split_dest, 'adv_%d_fxr.png' % (runner_id)), fxr, cmap='gray');

#    plt.imsave(join(split_dest, 'adv_%d_fx_crop.png'  % (runner_id)), fx_crop, cmap='gray');
#    plt.imsave(join(split_dest, 'adv_%d_fxr_crop.png' % (runner_id)), fxr_crop, cmap='gray');














