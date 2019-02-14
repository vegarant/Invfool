"""
The script performs the add more samples experiment shown in the paper.
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
from variational_tools import get_model_name_and_mask_name, get_uniform_sampling_patt, hand_f;

from adversarial_tools import compute_psnr;

import numpy as np
from mridata import VnMriReconstructionData
import mriutils
from os.path import join;

if __name__ == '__main__':
    

    image_config = 'configs/reco.yaml';

    # Specify which network you would like to load
    samp_type = 'unif'; 
    samp_frac = 0.15; 
    print('Using network with samp type: %s, samp frac: %g' % 
          (samp_type, samp_frac));
    model_name, mask_name = get_model_name_and_mask_name(samp_type, samp_frac);

    dest = 'results';
    data_config = tf.contrib.icg.utils.loadYaml(image_config, ['data_config']);

    output_name = join('results','image');  
    epoch = 1000; 

    checkpoint_config = tf.contrib.icg.utils.loadYaml('./configs/training.yaml',
                                                      ['checkpoint_config']);

    ckpt_dir = checkpoint_config['log_dir'] + '/' + model_name + '/checkpoints/'

    sess = tf.Session();

    try:
        # load from checkpoint if required
        epoch = vn.utils.loadCheckpoint(sess, ckpt_dir, epoch=epoch)
    except Exception as e:
        print(traceback.print_exc())

    # extract operators and variables from the graph
    u_op = tf.get_collection('u_op')[0]
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


    def val_fn(x_adj, kspace, coil_sens, mask, ref=np.zeros([1,640,368], dtype='float32')):

        u_i = sess.run(u_op, feed_dict={c_var[0] : coil_sens,
                                        m_var[0] : mask, 
                                        f_var[0] : kspace, 
                                        u_var[0] : x_adj,
                                        g_var[0] : ref});

        return u_i;


    subsampling_fact = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35,  0.5];
    nbr_center_frec = 28*np.ones(len(subsampling_fact), dtype='int64');
    nbr_center_frec[0] = 14; # For 5 % of the samples we can not sample 
                             # that many center frequencies
    selected_patients = [11,12,13,14,15,16,17,18,19,20];
    slice_nbr = 22;

    nbr_patients = len(selected_patients);
    psnr_vals = np.zeros([len(subsampling_fact), nbr_patients]);

    for k in range(nbr_patients):
        data_config['dataset']['patient'] = selected_patients[k];
        kspace, coil_sens, x_adj, ref, mask, norm \
                    = data.get_test_data(data_config['dataset'],
                                              data_config['dataset']['patient'],
                                              slice_nbr);
    
        ref_sq = np.squeeze(ref);
        for i in range(len(subsampling_fact)):
            mask = get_uniform_sampling_patt(subsampling_fact[i], 
                                         nbr_center_lines =nbr_center_frec[i]);
            mask = np.asarray([mask]);
            tot_samples = np.prod(mask.shape);
            nbr_samples = np.sum(mask);
            actual_subsampling = nbr_samples/tot_samples;

            x_adj = mriutils.mriAdjointOp(np.squeeze(kspace), 
                                      np.squeeze(coil_sens),
                                      mask).astype(np.complex64); 


            x_adj = np.asarray([x_adj]);
            
            u_T = val_fn(x_adj, kspace, coil_sens, mask);
        
            rec = np.squeeze(u_T);
            
            val = compute_psnr(rec, ref_sq);
            psnr_vals[i][k] = val;
    
    psnr_avrage = np.zeros(len(psnr_vals));
    for i in range(len(psnr_vals)):
        psnr_avrage[i] = np.mean(psnr_vals[i]); 

    lwidth = 2;
    fsize = 20;
    
    fig = plt.figure();
    plt.plot(subsampling_fact, psnr_avrage, "*-", linewidth = lwidth, ms = 12);
    p_min = min(psnr_avrage);
    p_max = max(psnr_avrage);
    plt.xlabel('Subsampling rate', fontsize=fsize);
    plt.ylabel('Avrage PSNR', fontsize=fsize);
#    plt.title('MRI-VN', fontsize=fsize);
    plt.plot([samp_frac, samp_frac], [0.98*p_min,1.02*p_max], 'r--', linewidth=lwidth);
    plt.axis([0, max(subsampling_fact), 0.98*p_min, 1.02*p_max]);
    if not os.path.isdir('plots_add_more_samples'):
        os.mkdir('plots_add_more_samples')    
    fig.savefig('plots_add_more_samples/mri_vn_psnr_add_more_samples.png',bbox_inches='tight');


