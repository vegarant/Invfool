from __future__ import print_function, division

import os;
import sys;
import time;
import numpy as np;
import theano;
import theano.tensor as T;
import lasagne;
import argparse;
import matplotlib.pyplot as plt;
import matplotlib.image as mpimg;


from os.path import join
import scipy.io;
from utils import compressed_sensing as cs
from utils.metric import complex_psnr
from cascadenet.network.model import build_d2_c2, build_d5_c5
from cascadenet.util.helpers import from_lasagne_format, to_lasagne_format;
from deep_mri_tools import *;
from deep_mri_config import network_path, mask_data_path, symbol_data_path;

# This scripts test the networks ability to recover a fine detailed structure. 

if __name__ == "__main__":
    sys.setrecursionlimit(2000);

    shuffle_batch = False;
    undersampling_rate = 4;

    batch_size = 2;
    Nx = 256;
    Ny = 256;

    input_shape = (batch_size, 2, Nx, Ny);

    A = scipy.io.loadmat(mask_data_path);
    mask = A['mask'];

    net_config, net = load_network(input_shape, network_path);

    val_fn = compile_f(net, net_config);
    f = lambda im: hand_f(val_fn, im, mask);    

    dest = 'plots_symbol';
    cmap = 'gray'

    fsize_list = [10];

    for j in range(len(fsize_list)): 
        fsize = fsize_list[j];
        fname = "deep_symb_fsize_%02d.mat" % (fsize);
        A = scipy.io.loadmat(join(symbol_data_path,fname));
        data = A['Y'];
        data1 = to_lasagne_format(data);
        pred = f(data1);
        pred = from_lasagne_format(pred);
        im = abs(pred[0,:,:]);
        fname = 'DeepMRINet_symb_rec_fsize_%02d.png' % (fsize);
        fname = join(dest, fname);
        mpimg.imsave(fname, im, cmap=cmap);






