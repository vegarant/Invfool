import tensorflow as tf;
import scipy;
import h5py
from os.path import join;
import _2fc_2cnv_1dcv_L1sparse_64x64_tanhrelu_upg as arch
import matplotlib.image as mpimg;
import matplotlib.pyplot as plt;
import numpy as np;
from automap_config import src_weights, src_mri_data, \
                           src_k_mask, src_automap_runner;
from adversarial_tools import scale_to_01
from pyfftw.interfaces import scipy_fftpack as fftw
from Runner import Runner;
from Automap_Runner import Automap_Runner;
import pickle;

def load_runner(run_nbr, src_runner=src_automap_runner):
    
    fname = join('data', 'runner_%d.pkl' % (run_nbr));
    fname = join(src_runner, fname);
    
    with open(fname, 'rb') as outfile:
            my_automap_runner = pickle.load(outfile);
    return my_automap_runner;
    
    



def sample_image(im, k_mask_idx1, k_mask_idx2):
    """
    Creates the fourier samples the AUTOMAP network is trained to recover.
    
    The parameters k_mask_idx1 and k_mask_idx2 cointains the row and column
    indices, respectively, of the samples the network is trained to recover.  
    It is assumed that these indices have the same ordering of the coefficents,
    as the network is used to recover. 

    :param im: Image, assumed of size [batch_size, height, width]. The intensity 
               values of the image should lie in the range [0, 1]. 
    :param k_maks_idx1: Row indices of the Fourier samples
    :param k_maks_idx2: Column indices of the Fourier samples

    :return: Fourier samples in the format the AUTOMAP network expect
    """

    # Scale the image to the right range 
    im1 = 4096*im;
    batch_size = im1.shape[0];
    nbr_samples = k_mask_idx1.shape[0];
    samp_batch = np.zeros([batch_size, 2*nbr_samples], dtype=np.float32);
    
    for i in range(batch_size):
        
        single_im = np.squeeze(im1[i,:,:]);
        fft_im = fftw.fft2(single_im);
        samples = fft_im[k_mask_idx1, k_mask_idx2];
        samples_real = np.real(samples);
        samples_imag = np.imag(np.conj(samples));
        samples_concat = np.squeeze(np.concatenate( (samples_real, samples_imag) ));

        samples_concat = ( 0.0075/(2*4096) )*samples_concat;
        samp_batch[i] = samples_concat;

    return samp_batch;

def adjoint_of_samples(samp_batch, k_mask_idx1, k_mask_idx2, N = 128):
    if len(samp_batch.shape) != 2:
        print('Warning: adjoint_of_samples -> samp_batch.shape is wrong')
    batch_size  = samp_batch.shape[0];
    nbr_samples = samp_batch.shape[1];
    samp_batch = ((2*4096)/0.0075)*samp_batch;
    
    adjoint_batch = np.zeros([batch_size, N, N], dtype=np.complex64);
    
    for i in range(batch_size):
        samples_concat = samp_batch[i];
        samples_real = samples_concat[:int(nbr_samples/2)];
        samples_imag = samples_concat[int(nbr_samples/2):];
        
        samples = samples_real - 1j*samples_imag;
        if len(samples.shape) == 1:
            samples = np.expand_dims(samples, axis=1);
        fft_im = np.zeros([N,N], dtype=samples.dtype);
        fft_im[k_mask_idx1, k_mask_idx2] = samples;

        adjoint = fftw.ifft2(fft_im)/4096;
        adjoint_batch[i, :, :] = adjoint;

    return adjoint_batch;

def hand_f(raw_f, x, k_mask_idx1, k_mask_idx2):
    """
    Takes in an image, subsample it in k-space, and reconstructs it using the 
    AUTOMAP network.

    :param f: Handle for the network reconstruction.
    :param x: Image of size [batch_size, height, width] with intensity values in the range [0, 1]
    :param k_mask_idx1: Row indices of k-space samples
    :param k_mask_idx2: Column indices of k-space samples
    
    :return: The reconstructed image.
    """

    samples = sample_image(x, k_mask_idx1, k_mask_idx2);
    im_rec = raw_f(samples);

    return im_rec;

def hand_dQ(raw_df, x, r, label, la, k_mask_idx1, k_mask_idx2):
    """
    Takes in an image and a perturbation (both in image domain), subsample 
    them in k-space, and computes the gradident of 
    
    Q(r) = ||f(A(x+r)) - f(Ax)||_{2}^{2} - (la/2)*||r||_{2}^{2}
  
    w.r.t. 'r'.

    :param raw_df: Handle for the networks gradient. The input to 
    :param x: Image of size [batch_size, height, width] with intensity values in the range [0, 1]
    :param r: Perturbation of size [height, width] 
    :param k_mask_idx1: Row indices of k-space samples
    :param k_mask_idx2: Column indices of k-space samples
    
    :return: The reconstructed image.
    """
    samples = sample_image(x+r, k_mask_idx1, k_mask_idx2);
    
    du_samples = raw_df(samples, label);
    adj_r = adjoint_of_samples(du_samples, k_mask_idx1, k_mask_idx2);
    
    dr1 = adj_r - la*r;
    return dr1;
    
    
def read_automap_weights(src_weights, fname_weights = 'CS_Poisson_For_Vegard.h5'):
    """
    Reads the automap weights from file, and return a h5py dictionary file 
    with the weights
    """
    f_weights = h5py.File(join(src_weights, fname_weights),  'r+')
    return f_weights;

def read_automap_k_space_mask(src_k_mask=src_k_mask, fname_k_mask_idx = 'k_mask_idx.mat'):
    """
    Reads the automap k_space indices, and return these.  
    """
    k_mask_idx_data = scipy.io.loadmat(join(src_k_mask, fname_k_mask_idx));
    idx1 = k_mask_idx_data['idx1'];
    idx2 = k_mask_idx_data['idx2'];
    return idx1, idx2;
   

def compile_network(sess, batch_size, src_weights=src_weights):
    
    f_weights = read_automap_weights(src_weights)
    
    W1_cnv = np.asarray(f_weights['W1_cnv']);
    W1_dcv = np.asarray(f_weights['W1_dcv']);
    W1_fc  = np.asarray(f_weights['W1_fc']);
    W2_cnv = np.asarray(f_weights['W2_cnv']);
    W2_fc  = np.asarray(f_weights['W2_fc']);
    b1_cnv = np.asarray(f_weights['b1_cnv']);
    b1_dcv = np.asarray(f_weights['b1_dcv']);
    b1_fc  = np.asarray(f_weights['b1_fc']);
    b2_cnv = np.asarray(f_weights['b2_cnv']);
    b2_fc  = np.asarray(f_weights['b2_fc']);
    
    model_in_vars =  ['W1_fc', 'b1_fc', 'W2_fc', 'b2_fc', 'W1_cnv', 'b1_cnv', 'W2_cnv', 'b2_cnv', 'W1_dcv', 'b1_dcv'];
    model_in_shape = [W1_fc.shape, b1_fc.shape, 
                      W2_fc.shape, b2_fc.shape, 
                      W1_cnv.shape, b1_cnv.shape, 
                      W2_cnv.shape, b2_cnv.shape,
                      W1_dcv.shape, b1_dcv.shape];

    precision = 'FP32';
    resolution = 128;
    in_dim = 19710; # (the resultant size of the k-space data after under sampling)
    h_dim = 25000;
    out_dim = 16384;

    net = arch.network(batch_size, precision, resolution, in_dim, h_dim, out_dim,
                       model_in_vars=model_in_vars, 
                       model_in_shapes=model_in_shape, 
                       trainable_model_in='False');


    sess.run(tf.global_variables_initializer(), feed_dict={
                             net['W1_fc']:  W1_fc,  net['b1_fc']:  b1_fc,
                             net['W2_fc']:  W2_fc,  net['b2_fc']:  b2_fc, 
                             net['W1_cnv']: W1_cnv, net['b1_cnv']: b1_cnv,
                             net['W2_cnv']: W2_cnv, net['b2_cnv']: b2_cnv, 
                             net['W1_dcv']: W1_dcv, net['b1_dcv']: b1_dcv})

    tf_ycrop = net['ycrop'];
    tf_x = net['x'];
    tf_label = tf.placeholder(tf.float32);
    tf_corrupt_prob = net['corrupt_prob'];
    
    tf_loss = tf.nn.l2_loss(tf_label - tf_ycrop);
    tf_grad = tf.gradients(tf_loss, tf_x);
    
    def raw_f(samples):
        corrupt_prob = np.asarray([0]);
        out = sess.run(tf_ycrop, feed_dict={tf_x: samples, 
                                      tf_corrupt_prob: corrupt_prob});
        return out;
    
    def raw_df(samples, label):
        corrupt_prob = np.asarray([0]);
        out = sess.run(tf_grad, feed_dict={tf_x: samples, 
                                           tf_corrupt_prob: corrupt_prob,
                                           tf_label: label});
        out = out[0];
        return out;

    return raw_f, raw_df;
