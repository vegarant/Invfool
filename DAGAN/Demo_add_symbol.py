import pickle
import tensorlayer as tl
import numpy as np
import os;
import os.path;
import nibabel as nib
import tensorflow as tf
import time
import matplotlib.pyplot as plt;

from model import *
from utils import *
from config import config, log_config
from scipy.io import loadmat, savemat
from dagan_tools import *;
from adversarial_tools import scale_to_01;
from os.path import join;

# This scripts subsample and reconstructs images with fine structures. 



def load_symbol_data(fsize_start, batch_size):
    fname = "dagan_symb_fsize_%02d.mat" % (fsize_start);
    fname = os.path.join(config.TEST.add_symbol_path, fname);
    data_dict = loadmat(fname);
    data = data_dict['Y1'];
    X = np.zeros([batch_size, *data.shape, 1], dtype='float');
    for i in range(batch_size):
        fname = "dagan_symb_fsize_%02d.mat" % (fsize_start+i);
        fname = os.path.join(config.TEST.add_symbol_path, fname);
        data_dict = loadmat(fname);
        data = data_dict['Y1'];
        data_0_1 = scale_to_01(data);
        data_1_1 = 2*data-1;

        X[i,:,:,0] = data_1_1;
    return X;
        
if __name__ == "__main__":

    # To test the network without refinement learning, change model name 
    # to 'unet'. 
    model_name = 'unet_refine'; # 'unet' or 'unet_refine'; 
    mask_name = 'gaussian1D';
    mask_perc = 20;
    rec_dest = 'plots_symbol/%s_%d_%s' % (model_name, mask_perc, mask_name);

    if not os.path.isdir('plots_symbol'):
        os.mkdir('plots_symbol');
        if not os.path.isdir(rec_dest):
            os.mkdir(rec_dest);

    fsize_start = 10;
    batch_size = 2;
    bd = 5; # Boundary between original and reconstruction

    mask = load_sampling_pattern(mask_name, mask_perc);

    batch = load_symbol_data(fsize_start, batch_size);

    nw, nh, nz = batch.shape[1:];

    tf_input = tf.placeholder('float32', [None, nw, nh, nz])
    network = load_network(model_name, tf_input);

    sess = load_session(model_name, mask_name, mask_perc, network);

    print('Compiling network function'); 
    def val_fn(x_adj):
        return sess.run(network.outputs, feed_dict={tf_input: x_adj});

    # Function handles
    f  = lambda x : hand_f(val_fn, x, mask);

    reconstructions = f(batch);

    for i in range(batch_size):
        im_rec = reconstructions[i,:,:,0];
        #im_rec_01 = scale_to_01(im_rec);
        fname = 'rec_%s_%d_fsize_%d.png' % (mask_name, mask_perc,
                                                    fsize_start+i);
        fname = join(rec_dest, fname);
        out_im = np.ones([nh, 2*nw+bd], dtype='float');
        out_im[:, 0:nw] = batch[i,:,:,0];
        out_im[:, nw+bd:] = im_rec;
        plt.imsave(fname, out_im, vmin=-1, vmax=1, cmap='gray');

