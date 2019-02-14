from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 
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
from variational_tools import get_model_name_and_mask_name;

import numpy as np
from mridata import VnMriReconstructionData
import mriutils
from os.path import join;

if __name__ == '__main__':

    image_config = 'configs/reco.yaml';

    samp_type = 'unif';
    samp_frac = 0.15; 
    print('Samp type: %s, samp frac: %g' % (samp_type, samp_frac));
    model_name, mask_name = get_model_name_and_mask_name(samp_type, samp_frac);
    
    dest = 'plots_symbol';
    if not os.path.isdir(dest):
        os.mkdir(dest);

    
    data_config = tf.contrib.icg.utils.loadYaml(image_config, ['data_config'])
    data_config['dataset']['mask'] = mask_name; # Set the relevant sampling 
                                                # pattern, it it is not correct

    epoch = 1000; 

    checkpoint_config = tf.contrib.icg.utils.loadYaml('./configs/training.yaml',
                                                      ['checkpoint_config']);

    ckpt_dir = checkpoint_config['log_dir'] + '/' + model_name + '/checkpoints/'

    with tf.Session() as sess:
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
                                       load_target=False)

        # run the model
        print('start reconstruction');

        dataset = 'coronal_pd_fs';
        patient = 21;
        slice1 = 21;
        feed_dict, norm = data.get_test_feed_dict(data_config['dataset'],
                                                  patient, 
                                                  slice1,
                                                  return_norm=True);

        # get the reconstruction, re-normalize and postprocesss it
        u_i = sess.run(u_op, feed_dict=feed_dict)
        

        patient_id = '%s-p%d-sl%d' % (dataset, patient, slice1);

        dest = os.path.join(dest, '%s_%g' % (samp_type, samp_frac));
        if not os.path.isdir(dest):
            os.mkdir(dest);

        fname = join(dest, 'slice_%d_can_u_see_it.png' % (slice1));
        print('Saving to %s' % (fname));
        plt.imsave(fname, np.squeeze(abs(u_i)), cmap='gray');



