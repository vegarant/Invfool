from __future__ import print_function, division

import sys;
from os.path import join
from cascadenet.util.helpers import from_lasagne_format
from cascadenet.util.helpers import to_lasagne_format
from utils import compressed_sensing as cs
from main_2d import iterate_minibatch, prep_input
import numpy as np;
from adversarial_tools import scale_to_01;
import scipy.misc;
from deep_mri_config import network_path, cardiac_data_path;
import scipy;
from numpy.lib.stride_tricks import as_strided

from adversarial_tools import compute_psnr; 

import Runner
from deep_mri_tools import *;

# This script test the networks ability to reconstruction images from different 
# subsampling rates. The sampling patterns are drawn at random, so be aware that
# this might affect the reconstruction quality, between conseceutive runs. 


def inc_cartesian_mask(shape, acc, old_indices=None, sample_n=10, centred=False):
    """
    This function is a slight modification of the function 
    utils.compressed_sensing.cartesian_mask. 
    It is modified so that it can remember which lines that where sampled in
    the previuous sampling process and include these lines in the next sampling
    pattern. It assumes that the sampling ratio `acc` is larger than the
    previuous `acc`. This function is only written for this spesific example,
    and it should be fixed if you would like to do something different.   
    """
    
    N, Nx, Ny = int(np.prod(shape[:-2])), shape[-2], shape[-1]
    pdf_x = cs.normal_pdf(Nx, 0.5/(Nx/10.)**2)
    lmda = Nx/(2.*acc)
    n_lines = int(Nx / acc)

    # add uniform distribution
    pdf_x += lmda * 1./Nx
    
    if sample_n and (old_indices is None):
        pdf_x[int(Nx/2-sample_n/2):int(Nx/2+sample_n/2)] = 0
        old_indices = np.arange(int(Nx/2-sample_n/2), int(Nx/2+sample_n/2));
        n_lines -= sample_n
    else: 
        pdf_x[old_indices] = 0;
        n_lines -= len(old_indices);
    pdf_x /= np.sum(pdf_x)
    mask = np.zeros((N, Nx))
   
    idx = np.random.choice(Nx, n_lines, False, pdf_x)
    
    idx = np.concatenate([idx, old_indices]);
    mask[:, idx] = 1

    if sample_n and (old_indices is None):
        mask[:, int(Nx/2-sample_n/2):int(Nx/2+sample_n/2)] = 1

    size = mask.itemsize
    mask = as_strided(mask, (N, Nx, Ny), (size * Nx, size, 0))

    mask = mask.reshape(shape)

    if not centred:
        mask = np.fft.ifftshift(mask, axes=(-1, -2))

    return mask, idx;

if __name__ == "__main__":
    dest = 'plots_more_samp';
    sys.setrecursionlimit(2000);
    
    batch_size = 30;
    shuffle_batch = False;
    
    data = load_data(cardiac_data_path);
    im = data[0:batch_size];
    print('data.shape: ', data.shape)

    Nt, Nx, Ny = data.shape;
    bdpx = 5;
    input_shape = (batch_size, 2, Nx, Ny);

    # Load and compile network
    net_config, net = load_network(input_shape, network_path);
   
    f = compile_f(net, net_config); # f(input_var, mask, k_space)

    us_rate = range(14,1,-1);
    n = len(us_rate);
    subsamp   = np.zeros(n);
    psnr_arr  = np.zeros(n);
    
    # Do first iteration outside loop;
    undersampling_rate = us_rate[0];
    mask, idx = inc_cartesian_mask((batch_size, Nx,Ny), undersampling_rate, sample_n=8);
    im_und, k_und = cs.undersample(im, mask, centred=False, norm='ortho');

    im_und_l = to_lasagne_format(im_und);
    k_und_l = to_lasagne_format(k_und);
    mask_l = to_lasagne_format(mask, mask=True);

    pred = f(im_und_l, mask_l, k_und_l);
    pred = from_lasagne_format(pred);
    #print("im.shape: ", im.shape);
    #print("pred.shape: ", pred.shape);
    psnr_values = np.zeros(batch_size);
    for i in range(batch_size):
        psnr_values[i] = compute_psnr(pred[i], im[i]);
    

    subsamp[0] = 1./undersampling_rate;
    psnr_arr[0] = np.mean(psnr_values);
    amask = np.squeeze(mask[0,:,:]);
    plt.imsave(os.path.join(dest, 'mask_data', "mask_k_%d.png" % 0), 
               amask, cmap='gray');
    
    for k in range(1,n): 
        undersampling_rate = us_rate[k];
        mask,idx = inc_cartesian_mask((batch_size, Nx, Ny), 
                                      undersampling_rate, 
                                      old_indices=idx,
                                      sample_n=8);
        
        
        #mask_name = os.path.join(dest, 'mask_data', 'mask_%d.mat' % (k));
        #scipy.io.savemat(mask_name, mdict={'mask': mask});
        im_und, k_und = cs.undersample(im, mask, centred=False, norm='ortho');
        #amask = np.squeeze(mask[0,:,:]);
        #plt.imsave(os.path.join(dest, 'mask_data', "mask_k_%d.png" % k), 
        #           amask, cmap='gray');

        im_und_l = to_lasagne_format(im_und);
        k_und_l = to_lasagne_format(k_und);
        mask_l = to_lasagne_format(mask, mask=True);

        pred = f(im_und_l, mask_l, k_und_l);
        pred = from_lasagne_format(pred);
        
        psnr_values = np.zeros(batch_size);
        for i in range(batch_size):
            psnr_values[i] = compute_psnr(pred[i], im[i]);
             
        subsamp[k] = 1./undersampling_rate;
        psnr_arr[k] = np.mean(psnr_values);


    psnr_name = os.path.join(dest, 'psnr_data', 'psrn_values.mat');
    scipy.io.savemat(psnr_name, mdict={'psnr_arr': psnr_arr});

    fsize=20;
    lwidth=2;
    
    fig = plt.figure();
    plt.plot(subsamp, psnr_arr, '*-', linewidth=lwidth, ms=12); 
    p_max = max(psnr_arr);
    p_min = min(psnr_arr);
    x_bar = 1/3;
    plt.plot([x_bar, x_bar], [0.98*p_min, 1.02*p_max], 'r--', linewidth=lwidth);
    plt.axis([0, max(subsamp), 0.98*p_min, 1.02*p_max]);
    plt.xlabel('Subsampling rate', fontsize=fsize); 
    plt.ylabel('Avrage PSNR', fontsize=fsize);
    #plt.title('Deep MRI net', fontsize=fsize);
    fname_plot = os.path.join(dest, 'deep_MRI_add_more_samples.png');
    fig.savefig(fname_plot, bbox_inches='tight');
    #plt.show();
