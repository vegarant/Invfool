# This file loads the runner object created by Demo_adversarial_perturbation.py
# and create the actual images. Make sure that you set the right Runner ID. 

import sys;
from os.path import join
from cascadenet.util.helpers import from_lasagne_format
from cascadenet.util.helpers import to_lasagne_format
from utils import compressed_sensing as cs
from main_2d import iterate_minibatch, prep_input
import numpy as np;
from Deep_MRI_Runner import Deep_MRI_Runner;
from deep_mri_tools import *;
from adversarial_tools import *;
import matplotlib.image as mpimg;

from deep_mri_config import cardiac_data_path, network_path; 


if __name__ == "__main__":

    dest = 'plots_adv'
    runner_id = 24; # Make sure you load the right runner object.
    runner = load_runner(runner_id);
    runner.__class__ = Deep_MRI_Runner;

    print('perp_start: %s'    % runner.perp_start);
    print('ps_factor:  %g'    % runner.perp_start_factor);
    print('lambda:     %g'    % runner.la);
    print('max_itr:    %g'    % runner.max_itr);
    print('warm_start: %s'    % runner.warm_start);
    print('ws_factor:  %g'    % runner.warm_start_factor);
    print('Momentum:      %g' % runner.momentum);
    print('Learning rate: %g' % runner.learning_rate);
    print('max_r_norm:    %g' % runner.max_r_norm);
    print('max_diff_norm: %g' % runner.max_diff_norm);
    

    x0 = runner.x0[0];
#    x0 = np.abs(from_lasagne_format(x0));
#    x0 = to_lasagne_format(x0);


    mask = runner.mask[0];
    input_shape = x0.shape;

    net_config, net = load_network(input_shape, network_path);
    val_fn, df = compile_functions(net, net_config);

    f = lambda im, : hand_f(val_fn, im, mask);    

    split_folder = os.path.join(dest, 'splits');
    if not os.path.isdir(dest):
        os.mkdir(dest);
        if not os.path.isdir(split_folder):
            os.mkdir(split_folder);
    
    im_nbr = 0;
    max_itr = runner.max_itr;
    N = input_shape[-1];
    bd = 5;
    fID = open(os.path.join(dest, 'description.txt'), 'w');

    # Make first perturbation equal to 0;
    a = np.zeros(runner.r[0].shape, dtype= runner.r[0].dtype);
    perturbation_list = list(runner.r);
    perturbation_list[0] = a; # Remove our initial guess perturbation r_0.
    for i in range(len(runner.r)):
        #r = to_lasagne_format(np.real(from_lasagne_format(runner.r[i])));
        r = perturbation_list[i];

        fx  = f(x0);
        fxr = f(x0 + r);


        x_i1   = from_lasagne_format(x0);
        r_i1   = from_lasagne_format(r);
        fx_i1  = from_lasagne_format(fx);
        fxr_i1 = from_lasagne_format(fxr);

        x_i = x_i1[im_nbr];
        r_i = r_i1[im_nbr];
        xpr = x_i+r_i;
        fx_i = fx_i1[im_nbr];
        fxr_i = fxr_i1[im_nbr];

        nx = l2_norm_of_tensor(x_i);
        nr = l2_norm_of_tensor(r_i);
        nf_diff = l2_norm_of_tensor(fx_i-fxr_i);

        next_str = '%2d: |f(x)-f(x+r)|: %g, |r|: %g, |f(x)-f(x+r)|/|r|: %g |r|/|x|: %g' % \
              (i, abs(nf_diff), nr, abs(nf_diff)/nr, nr/nx );
        print(next_str);
        fID.write(next_str + '\n');

        imax = np.amax(np.abs(x_i));
        im_concat = np.ones([2*N+bd,2*N+bd]);
        im_concat[0:N, 0:N] = abs(x_i);
        im_concat[0:N, N+bd:] = abs(xpr);
        im_concat[N+bd:, 0:N] = abs(fx_i);
        im_concat[N+bd:, N+bd:] = abs(fxr_i);

        itr = i*max_itr;
        fname = os.path.join(dest, 'r_%d_adv_itr_%04d.png' % (runner_id, itr));
        mpimg.imsave(fname, im_concat, vmin=0, vmax=imax, cmap='gray');

        fname_r = os.path.join(split_folder, 'r_%d_adv_xpr_itr_%04d.png' % (runner_id, itr));
        fname_fxr = os.path.join(split_folder, 'r_%d_adv_fxpr_itr_%04d.png' % (runner_id, itr));

        mpimg.imsave(fname_r, abs(xpr), vmin=0, vmax=imax, cmap='gray');
        mpimg.imsave(fname_fxr, abs(fxr_i), vmin=0, vmax=imax, cmap='gray');

    fID.close(); 

    # Export data so that we can perform CS reconstruction.
    r_idx = 12; # Perturbation we wish to export.
    runner.export_data(dest+'/runner_%d_' % (runner_id), r_idx);














