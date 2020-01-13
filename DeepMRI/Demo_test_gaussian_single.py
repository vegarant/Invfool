"""
This script reads the different carefully constructed perturbations r_1, r_2
and r_3, add gaussian noise to them, and reconstruction the result using the
Deep MRI network.
"""

from cascadenet.util.helpers import from_lasagne_format
from cascadenet.util.helpers import to_lasagne_format
from utils import compressed_sensing as cs
from main_2d import iterate_minibatch, prep_input
import numpy as np;
import sys
import os
from os.path import join
from scipy.io import loadmat, savemat
from deep_mri_config import network_path, deep_mri_runner_path; 
from Deep_MRI_Runner import Deep_MRI_Runner;
from deep_mri_tools import load_runner, load_network, compile_f;
from adversarial_tools import l2_norm_of_tensor; 
import matplotlib.image as mpimg;

if __name__ == "__main__":
    plot_fldr = 'plots_gauss_single'
    sys.setrecursionlimit(2000)

    if not os.path.isdir(plot_fldr):
        os.mkdir(plot_fldr)

    N = 256
    runner_id = 24
    factor = 0.25;
    runner = load_runner(runner_id)

    im_nbr = 0;

    x0 = runner.x0[0]; # shape [2,2,256,256]
    mask = runner.mask[0] # shape [2, 256, 256]
    r_list = runner.r; # length 16
    # runner.max_itr is 500

    r1 = r_list[4]
    r2 = r_list[8]
    r3 = r_list[12]

    norm_r1 = l2_norm_of_tensor(r1); 
    norm_r2 = l2_norm_of_tensor(r2); 
    norm_r3 = l2_norm_of_tensor(r3); 

    print(norm_r2/norm_r1)


    batch_size = x0.shape[0] # 2
    input_shape = [batch_size, 2, N, N]

    # Load and compile network
    net_config, net = load_network(input_shape, network_path);
   
    f = compile_f(net, net_config); # f(input_var, mask, k_space)

    noise_raw = np.random.normal(0,1, size=r_list[0].shape);
    noise_raw_complex = from_lasagne_format(noise_raw);
    noise_und, k_und = cs.undersample(noise_raw_complex, mask, centred=False, norm='ortho');


    norm_noise_und = l2_norm_of_tensor(noise_und);
    scaled_noise1 = (factor*norm_r1*noise_und)/norm_noise_und
    scaled_noise2 = (factor*norm_r2*noise_und)/norm_noise_und
    scaled_noise3 = (factor*norm_r3*noise_und)/norm_noise_und

    data = from_lasagne_format(x0);
    r1_no_l = from_lasagne_format(r1);
    r2_no_l = from_lasagne_format(r2);
    r3_no_l = from_lasagne_format(r3);
    im1_noisy = data + r1_no_l + scaled_noise1
    im2_noisy = data + r2_no_l + scaled_noise2
    im3_noisy = data + r3_no_l + scaled_noise3

    im_und1, k_und1 = cs.undersample(im1_noisy, mask, centred=False, norm='ortho');
    im_und2, k_und2 = cs.undersample(im2_noisy, mask, centred=False, norm='ortho');
    im_und3, k_und3 = cs.undersample(im3_noisy, mask, centred=False, norm='ortho');

    mask_l = to_lasagne_format(mask, mask=True);
    
    im_und_l1 = to_lasagne_format(im_und1);
    k_und_l1  = to_lasagne_format(k_und1);
    im_und_l2 = to_lasagne_format(im_und2);
    k_und_l2  = to_lasagne_format(k_und2);
    im_und_l3 = to_lasagne_format(im_und3);
    k_und_l3  = to_lasagne_format(k_und3);

    pred1 = f(im_und_l1, mask_l, k_und_l1);
    pred1 = from_lasagne_format(pred1);
    pred2 = f(im_und_l2, mask_l, k_und_l2);
    pred2 = from_lasagne_format(pred2);
    pred3 = f(im_und_l3, mask_l, k_und_l3);
    pred3 = from_lasagne_format(pred3);

    
    fname_im1 = 'im_noise1_r_fact_%4.2f.png' % (factor)
    fname_im2 = 'im_noise2_r_fact_%4.2f.png' % (factor)
    fname_im3 = 'im_noise3_r_fact_%4.2f.png' % (factor)
    fname_rec1 = 'im_rec_noise1_r_fact_%4.2f.png' % (factor)
    fname_rec2 = 'im_rec_noise2_r_fact_%4.2f.png' % (factor)
    fname_rec3 = 'im_rec_noise3_r_fact_%4.2f.png' % (factor)


    mpimg.imsave(join(plot_fldr, fname_im1), abs(np.squeeze(im1_noisy[im_nbr,:,:])), cmap='gray');
    mpimg.imsave(join(plot_fldr, fname_im2), abs(np.squeeze(im2_noisy[im_nbr,:,:])), cmap='gray');
    mpimg.imsave(join(plot_fldr, fname_im3), abs(np.squeeze(im3_noisy[im_nbr,:,:])), cmap='gray');

    mpimg.imsave(join(plot_fldr, fname_rec1), abs(np.squeeze(pred1[im_nbr,:,:])), cmap='gray');
    mpimg.imsave(join(plot_fldr, fname_rec2), abs(np.squeeze(pred2[im_nbr,:,:])), cmap='gray');
    mpimg.imsave(join(plot_fldr, fname_rec3), abs(np.squeeze(pred3[im_nbr,:,:])), cmap='gray');
    
    fname_data = 'data_gauss_noise_p_r_factor_%4.2f.mat' % (factor);
    savemat(join(plot_fldr, fname_data), 
        {'r1': r1_no_l,
         'r2': r2_no_l,
         'r3': r3_no_l,
         'scaled_noise1': scaled_noise1,
         'scaled_noise2': scaled_noise2,
         'scaled_noise3': scaled_noise3})
    
    


















