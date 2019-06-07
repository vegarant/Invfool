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
from deep_mri_config import cardiac_data_path, network_path; 

def display_perturbation(runner, f, im_nbr=0, cmap='gray'):

    r = runner.r[-1];
    batch = runner.x0[-1];
#    mask = runner.mask[0];

    fx  = f(batch);
    fxr = f(batch+r);

    x_i1   = from_lasagne_format(batch);
    r_i1   = from_lasagne_format(r);
    fx_i1  = from_lasagne_format(fx);
    fxr_i1 = from_lasagne_format(fxr);

    x_i = x_i1[im_nbr];
    r_i = r_i1[im_nbr];
    fx_i = fx_i1[im_nbr];
    fxr_i = fxr_i1[im_nbr];

    nx = l2_norm_of_tensor(x_i);
    nr = l2_norm_of_tensor(r_i);
    nf_diff = l2_norm_of_tensor(fx_i-fxr_i);

    print('|f(x)-f(x+r)|: %g, |r|: %g, |f(x)-f(x+r)|/|r|: %g |r|/|x|: %g' % \
          (abs(nf_diff), nr, abs(nf_diff)/nr, nr/nx ));

    plt.subplot(231); plt.matshow(abs(x_i), cmap=cmap, fignum=False);
    plt.title('|x|'); plt.colorbar();

    plt.subplot(232); plt.matshow(abs(x_i+r_i), cmap=cmap, fignum=False);
    plt.title('|x+r|'); plt.colorbar();

    plt.subplot(233); plt.matshow(abs(r_i), cmap=cmap, fignum=False);
    plt.title('|r|'); plt.colorbar();

    plt.subplot(234); plt.matshow(abs(fx_i), cmap=cmap, fignum=False);
    plt.title('|f(x)|'); plt.colorbar();

    plt.subplot(235); plt.matshow(abs(fxr_i), cmap=cmap, fignum=False);
    plt.title('|f(x+r)|'); plt.colorbar();

    plt.subplot(236); plt.matshow(abs(fxr_i-fx_i), cmap=cmap, fignum=False);
    plt.title('|f(x+r)-f(x)|'); plt.colorbar();
    plt.show(block=True);


if __name__ == "__main__":

    sys.setrecursionlimit(2000);

    data_path= cardiac_data_path; #'/home/va304/software/mri-rec/data/cardiac.mat'; 
    #network_path= network_path'/home/va304/software/mri-rec/models/pretrained/d5_c5.npz';
    batch_size = 2;
    shuffle_batch = False;
    undersampling_rate = 3;


    # Optimization parameters
    max_itr = 500;
    max_r_norm = float('Inf');
    max_diff_norm = float('Inf');
    la = 0.001;
    warm_start = 'off';
    warm_start_factor = 0;
    perp_start = 'rand';
    perp_start_factor = 0.01;
    momentum = 0.9;
    learning_rate = 0.01;
    verbose=True; 


    data = load_data();
    #batch_size = data.shape[0];
    Nt, Nx, Ny = data.shape;
    input_shape = (batch_size, 2, Nx, Ny);

    mask = cs.cartesian_mask((batch_size, Nx, Ny), 
                              undersampling_rate, sample_n=8)

#    mask = np.zeros([2,2,Nx,Ny]).astype('float32');
#    mask[:,0,:,:]= dummy_mask;
#    mask[:,1,:,:]= dummy_mask;
#
#    runner_id = 25;
#    runner = load_runner(runner_id);
#    fname = 'data/data_runner_%d_' % (runner_id);
#    runner.export_data(fname);
#    
#    mask = runner.mask[0];

    net_config, net = load_network(input_shape, network_path);
    val_fn, df = compile_functions(net, net_config);

    f = lambda im, : hand_f(val_fn, im, mask);    
    dQ = lambda im, r,  pred, la: hand_dQ(df,im,r, mask, pred, la);
    data1 = to_lasagne_format(data);
    datab = data1[0:batch_size];

    runner = Deep_MRI_Runner(max_itr, max_r_norm, max_diff_norm, 
                           la=la, 
                           warm_start=warm_start,
                           warm_start_factor=warm_start_factor,
                           perp_start=perp_start,
                           perp_start_factor=perp_start_factor,
                           momentum=momentum,
                           learning_rate= learning_rate,
                           verbose=verbose,
                           mask= [mask]
                           );

    for i in range(15):
        print "\n\n\ni: %d\n\n\n" % (i);
        runner.find_adversarial_perturbation(f, dQ, datab, only_real=False);

    runner_ID = runner.save_runner(f);
    print('Runner ID: %d' % runner_ID);



#    display_perturbation(runner, f, im_nbr=1, cmap='gray')






