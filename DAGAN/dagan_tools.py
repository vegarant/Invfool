

import tensorflow as tf;
import tensorlayer as tl;
import numpy as np;
import os;
import Runner;
from os.path import join;
from model import u_net_bn;
from utils import *;

from scipy.io import loadmat, savemat
from config import config, log_config


def load_test_data():
    testing_data_path = config.TRAIN.testing_data_path
    with open(testing_data_path, 'rb') as f:
        X_test = pickle.load(f)
    return X_test;

def load_network(model_name, tf_input):
    # define generator network
    if model_name == 'unet':
        network = u_net_bn(tf_input, is_train=False, reuse=True, is_refine=False)
    elif model_name == 'unet_refine':
        network = u_net_bn(tf_input, is_train=False, reuse=True, is_refine=True)
    else:
        raise Exception("unknown model")
    return network;

def load_sampling_pattern(mask_name, mask_perc):
    
    if mask_name.lower() == "gaussian2d".lower():
        mask = \
            loadmat(
                os.path.join(config.TRAIN.mask_Gaussian2D_path, "GaussianDistribution2DMask_{}.mat".format(mask_perc)))[
                'maskRS2']
    elif mask_name.lower() == "gaussian1d".lower():
        mask = \
            loadmat(
                os.path.join(config.TRAIN.mask_Gaussian1D_path, "GaussianDistribution1DMask_{}.mat".format(mask_perc)))[
                'maskRS1']
    elif mask_name.lower() == "poisson2d".lower():
        mask = \
            loadmat(
                os.path.join(config.TRAIN.mask_Gaussian1D_path, "PoissonDistributionMask_{}.mat".format(mask_perc)))[
                'population_matrix']
    else:
        raise ValueError("no such mask exists: {}".format(mask_name))
    return mask;



def load_session(model_name, mask_name, mask_perc, network):
    model_name = model_name.lower()
    mask_name = mask_name.lower()
    checkpoint_dir = os.path.join(config.TEST.checkpoint_path, "checkpoint_{}_{}_{}".format(model_name, mask_name, mask_perc));
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    tl.layers.initialize_global_variables(sess);

    # load generator and discriminator weights (for continuous training purpose)
    tl.files.load_and_assign_npz(sess=sess,
                                 name=os.path.join(checkpoint_dir, model_name) + '.npz',
                                 network=network);
    return sess;


def load_runner(ID, data_path='/local/scratch/public/va304/dagan/runners/data'):
    """
    Loades a runner object. 
    
    This function should be updated
    """
    fname = 'runner_%d.npz' % (ID);
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
                        data_path='/local/scratch/public/va304/deep_mri'):
    
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
    
def load_data2(batch_size, data_path='knee1.png'):
    im = plt.imread(data_path);
    data = np.zeros([batch_size, im.shape[0], im.shape[1]], dtype=im.dtype);
    for i in range(batch_size):
        data[i, :,:] = im;
    return data;

def load_text_data(fsize, color = 'red'):
    """
    Load the 'can u read me?' images.
    """
    data_path = '/local/scratch/public/va304/deep_mri/data_mat';
    fname = 'data_cardiac_%s_fsize_%d.mat' % (color, fsize);
    A = sio.loadmat(join(data_path, fname));
    data = A['Y'];
    return data;


def hand_f(val_fn, x, mask):
    # x has shape [batch_size, height, widht, channels]
    

    x_adj = threading_data(x, fn=to_bad_img, mask=mask);
    return val_fn(x_adj);

def hand_dQ(val_df, x, r, pred, lambda1, mask):
    
    # Sample image and perturbation.
    #x_adj = Adjoint_Sampling(Sampling(x, mask), mask);
    #print('x.shape:', x.shape);
    x_adj = threading_data(x, fn=to_bad_img, mask=mask);
    #r_adj = Adjoint_Sampling(Sampling(r, mask), mask);
    r_adj = threading_data(r, fn=to_bad_img, mask=mask);    

    # This is the direct gradient of the network
    #print('x_adj.shape:', x_adj.shape);
    df = val_df(x_adj+r_adj, pred);
    
    # Performing the last two steps of the backpropagation by hand 
    #df = Adjoint_Sampling(Sampling(df, mask), mask);
    df = threading_data(df[0], fn=to_bad_img, mask=mask);
    return df - lambda1*r;




