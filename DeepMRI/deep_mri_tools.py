#!/usr/bin/env python
from __future__ import print_function, division

import os
import time
import numpy as np
import theano
import theano.tensor as T
import lasagne
import argparse
import matplotlib.pyplot as plt
import Runner;

from os.path import join
import scipy.io as sio;

from utils import compressed_sensing as cs
from utils.metric import complex_psnr

from cascadenet.network.model import build_d2_c2, build_d5_c5
from cascadenet.util.helpers import from_lasagne_format, to_lasagne_format;

from deep_mri_config import deep_mri_runner_path, cardiac_data_path, network_path; 

# Note the Deep MRI directory must be in your python path
#l2_norm_of_tensor = lambda x: (abs(x)**2).sum();

def hand_dQ(df, im, r, mask, pred, la):

    imr = from_lasagne_format(im+r);
    im_und, k_und = cs.undersample(imr, mask, centred=False, norm='ortho')

    im_und_l = to_lasagne_format(im_und)
    k_und_l = to_lasagne_format(k_und)
    mask_l = to_lasagne_format(mask, mask=True)

    dr = df(im_und_l, mask_l, k_und_l, pred);
    dr = dr - la*r;

    return dr;

def hand_dQ_real(df, im, r, mask, pred, la):

    imr = from_lasagne_format(im+r);
    im_und, k_und = cs.undersample(imr, mask, centred=False, norm='ortho')
    
    im_und_l = to_lasagne_format(im_und)
    k_und_l = to_lasagne_format(k_und)
    mask_l = to_lasagne_format(mask, mask=True)

    #dr = df(im_und_l, mask_l, k_und_l, pred);
    dr = df(im_und_l, mask_l, k_und_l, pred);
    dr = dr - la*r;
    dr[:,1::2, ...] = 0;
    return dr;

def hand_f(f, im, mask):
    im = from_lasagne_format(im);
    im_und, k_und = cs.undersample(im, mask, centred=False, norm='ortho')
    
    im_und_l = to_lasagne_format(im_und)
    k_und_l = to_lasagne_format(k_und)
    mask_l = to_lasagne_format(mask, mask=True)
    
    pred = f(im_und_l, mask_l, k_und_l);
    return pred;

def load_runner(ID, data_path=deep_mri_runner_path):
    fname = 'data/runner_%d.npz' % (ID);
    fname = join(data_path, fname);
    data = np.load(fname);
    
    lr = float(data['lr']);
    momentum = float(data['momentum']);
    smoothing_eps = float(data['smoothing_eps']);
    la = float(data['la']);
    v = data['v'];
    r = data['r'];
    x0 = data['x0'];
    mask = data['mask'];
    optimizer = str(data['optimizer']);
    backlog = str(data['backlog']);
    max_itr = int(data['max_itr']);
    max_r_norm = float(data['max_r_norm']);
    max_diff_norm = float(data['max_diff_norm']);
    ps = str(data['ps']);
    psf = float(data['psf']);
    ws = str(data['ws']);
    wsf = float(data['wsf']);
    
    length_r = r.shape[0];
    r_list = [];
    v_list = [];
    for i in range(length_r):
        r_list.append(r[i]);
        v_list.append(v[i]);
    
    runner = Runner.Runner(max_itr, 
                 max_r_norm,
                 max_diff_norm,
                 la=la, 
                 warm_start = ws, # ||f(x+r) - f(x) + p||_{2}^{2} 
                 warm_start_factor = wsf,
                 perp_start = ps,
                 perp_start_factor=psf, 
                 optimizer=optimizer,
                 momentum = momentum,
                 smoothing_eps = smoothing_eps,
                 learning_rate = lr);
    runner.backlog = backlog;
    runner.v = v_list;
    runner.r = r_list;
    runner.x0 = [x0];
    runner.mask = [mask];
    return runner;

def convert_runner_to_matlab_format(runner_id, 
                        data_path=deep_mri_runner_path):
    
    fname = join(data_path, 'data', 'runner_%d.npz' % runner_id);
    fname_out = join(data_path, 'data_mat', 'runner_%d.mat' % runner_id);
    data  = np.load(fname);
    
    out_dict = {}
    for key in data.keys():
        out_dict[key] = data[key];
    
    r1 = out_dict['r'];
    n, batch_size, ch, Ny, Nx = r1.shape;

    r = np.zeros([n, batch_size, Ny, Nx], dtype='complex64');
    for i in range(n):
        r[i] = from_lasagne_format(r1[i]);

    out_dict['x0'] = from_lasagne_format(data['x0']); 
    out_dict['r']  = r;

    sio.savemat(fname_out, out_dict)
    

def load_network(input_shape, 
                 network_path=network_path):
    """
    Loads the pretrained network, specified by the absolute_path.

    :param input_shape:   - (batch_size, 2, Nx, Ny)
    :param network_path: - path to network weights
    
    :returns: net_config, network
    """
    
    #input_shape = (batch_size, 2, Nx, Ny)
    #net_config, net,  = build_d2_c2(input_shape)

    # Load D5-C5 with pretrained params
    net_config, network,  = build_d5_c5(input_shape)
    # D5-C5 with pre-trained parameters
    with np.load(network_path) as f:
        param_values = [f['arr_{0}'.format(i)] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(network, param_values)

    return net_config, network;

def load_data(data_path=cardiac_data_path):
    
    data = sio.loadmat(data_path)['seq']
    nx, ny, nt = data.shape
    test = np.transpose(data, (2, 0, 1))

    return test;

def compile_f(network, net_config):
    """
    Compile the neural network f.
    """
    # Theano variables
    input_var = net_config['input'].input_var;
    mask_var = net_config['mask'].input_var;
    kspace_var = net_config['kspace_input'].input_var;
    target_var = T.tensor4('targets');

    # Objective
    pred = lasagne.layers.get_output(network)

    print(' Compiling ... ')
    t_start = time.time()

    val_fn = theano.function([input_var, mask_var, kspace_var],
                             pred,
                             on_unused_input='ignore')

    t_end = time.time()
    print(' ... Done, took %.4f s' % (t_end - t_start))
    return val_fn;

def compile_functions(network, net_config):
    """
    Compile the functions f and dQ.
    """
    # Theano variables
    input_var = net_config['input'].input_var;
    mask_var = net_config['mask'].input_var;
    kspace_var = net_config['kspace_input'].input_var;
    target_var = T.tensor4('targets');

    # Objective
    pred = lasagne.layers.get_output(network)
    # complex valued signal has 2 channels, which counts as 1.
    loss_sq = lasagne.objectives.squared_error(target_var, pred).mean() * 2
    
    print(' Compiling ... ')
    t_start = time.time()

    val_fn = theano.function([input_var, mask_var, kspace_var],
                             pred,
                             on_unused_input='ignore')
    
    df_grad = T.grad(loss_sq, input_var);
    
    df = theano.function([input_var, mask_var, kspace_var, target_var], df_grad);
    
    t_end = time.time()
    
    print(' ... Done, took %.4f s' % (t_end - t_start))

    return val_fn, df;

def full_sampling(shape):
    mask = np.ones(shape);
    return mask;    

    
    
if __name__ == "__main__":
    pertub_SGA(1,2,3,4,5,6, verbose = False, max_r_norm = 4, max_diff_norm = 500.1);
    




