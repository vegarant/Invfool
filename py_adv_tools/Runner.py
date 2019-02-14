import numpy as np;
from adversarial_tools import pertub_SGA, l2_norm_of_tensor
import matplotlib.pyplot as plt
from os.path import join
import scipy.io
import sys;

class Runner:
    """ Class keeping track of each search for adversarial perturbations.
    
    The class contains all relevant parameters, and can be used to initialize 
    gradient search algorithms.
    """
    
    def __init__(self, 
                 max_itr, 
                 max_r_norm,
                 max_diff_norm,
                 la=-1, 
                 warm_start = 'off', # ||f(x+r) - f(x) + p||_{2}^{2} 
                 warm_start_factor = 0,
                 perp_start = 'rand',
                 perp_start_factor=1, 
                 reference='pred',
                 optimizer='SGA',
                 momentum = 0.9,
                 smoothing_eps = 1e-8,
                 learning_rate = 0.01,
                 verbose=True,
                 mask=None
                 ):
        """
    Let f be a neural network and let y = A x where x is the original signal and
    y is some measurements obtained via the measurement matrix A. Let 
    Q(r) = ||f(y + Ar) - reference + p||_{2}^{2} - la*||r||_{2}^{2}.
    
    :param max_itr: Maximum number of iterations.
    :param max_r_norm: Stop if ||r||_{2} > max_r_norm.
    :param max_diff_norm: Stop if ||f(y+r) - reference||_{2} > max_r_norm.
    :param la: la, see def of Q.
    :param warm_start: 'rand', values of p are drawn from uniformly at random from [0,1],
                       'randn', values of p are drawn from a Gaussian distribution with
                                mean 0 and standard deviation 1,
                       'ones', all values of p equals 1,
                       'off', all values of p equals 0.
    :param warm_start_factor: p = warm_start_factor*p.
    :param perp_start: 'rand', values of r are drawn from uniformly at random from [0,1],
                       'randn', values of r are drawn from a Gaussian distribution with
                                mean 0 and standard deviation 1,
                       'ones', all values of r equals 1,
                       'off', all values of r equals 0.
    :param perp_start_factor: r = perp_start_factor*r.
    :param reference: The reference image the gradient will be computed with 
                      respect to. It has the following options.
                        - 'pred' The output of the network i.e. f(y).
                        - 'true' The true image 'x'
                        - Numpy array: A spesific value.
    :param optimizer: 'SGA' call pertub_SGA
    :param momentum:  momentum used by the optimizer.
    :param smoothing_eps:  smoothing_eps used by the optimizer. (Not supported yet).
    :param learning_rate:  learning rate used by the optimizer.
    :param verbose: Whether or not to print information.
    :param mask: Sampling mask used for Fourier sampling.
        """
        self.verbose = verbose; 
        self.max_itr = max_itr;
        self.max_r_norm = max_r_norm;
        self.max_diff_norm = max_diff_norm;
        self.la = la;
        self.warm_start = warm_start; 
        self.warm_start_factor = warm_start_factor;
        self.perp_start = perp_start;
        self.perp_start_factor = perp_start_factor; 
        self.reference = reference;
        self.optimizer = optimizer;
        self.momentum = momentum;
        self.smoothing_eps = smoothing_eps;
        self.learning_rate = learning_rate;
        self.mask = mask;
        self.r = [];  # List of perturbations fund so far
        self.v = [];   
        self.x0 = []; # Images the perturbations should be applied to
        self.backlog = '';

    def __str__(self):
        ref_string = self.reference;
        if type(ref_string) is np.ndarray:
            ref_string = 'ndarray';
        mask = self.mask;
        mask_str = '';
        if type(mask) == list:
            if mask: # mask is non-empty
                if type(mask[0]) == np.ndarray:
                    mask_str = str(mask[0].shape);

            


        return """
max_itr:       %g
max_r_norm:    %g
max_diff_norm: %g 
la:            %g
warm_start:    %s
warm_start_factor: %g
perp_start:        %s
perp_start_factor: %g
reference:         %s
optimizer:         %s
momentum:          %g
learning_rate:     %g
mask.shape:        %s
len(r):            %d
len(v):            %d
len(x0):           %d
""" % ( self.max_itr,
        self.max_r_norm,
        self.max_diff_norm,
        self.la,
        self.warm_start,
        self.warm_start_factor,
        self.perp_start,
        self.perp_start_factor,
        ref_string,
        self.optimizer,
        self.momentum,
        self.learning_rate,
        mask_str,
        len(self.r), 
        len(self.v), 
        len(self.x0));


    
    def _read_count(self, count_path):
        """ Read and updates the runner count. 
        
        To keep track of all the different runs of the algorithm, one store the 
        run number in the file 'COUNT.txt' at ``count_path``. It is assumed that 
        the file 'COUNT.txt' is a text file containing one line with a single 
        integer, representing number of runs so far. 

        This function reads the current number in this file and increases the 
        number by 1. 
        
        :return: Current run number (int).
        """
        fname = join(count_path, 'COUNT.txt');
        infile = open(fname);
        data = infile.read();
        count = int(eval(data));
        infile.close();

        outfile = open(fname, 'w');
        outfile.write('%d ' % (count+1));
        outfile.close();
        return count;


    def find_adversarial_perturbation(self, f, dQ, batch):
        """ Search for adversarial perturbation.

        :param f: Neural network. This is a function handle which should take 
                  as input the ground truth batch, sample this into a k-space 
                  image, and feeds the subsampled image to the network. The 
                  output from the function handle should be the network 
                  reconstruction of the entire batch.
        :param dQ: Gradient w.r.t. r for the function Q(r) = ||f(Ax+Ar)-f(Ax)||_{2}^{2} - la*||r||_{2}^{2}.
                   The function dQ is a function handle taking the inputs (image, r, f(Ax), la). 
        :param batch: Batch of images i.e. x.

        :return: Nothing, but at the end of the lists ``self.r`` and ``self.v`` 
                  the adversarial perturbation and the intermediate step v, 
                  are added. The ``self.backlog`` attribute is also updated. 

        """

        ref_string = self.reference;
        if type(ref_string) is np.ndarray:
            ref_string = 'ndarray';
        if (self.verbose):
            print('------------------------------------');
            print('Running all images with paramters:');
            print('Optimizer:  %s' % self.optimizer);
            print('perp_start: %s' % self.perp_start);
            print('ps_factor:  %g' % self.perp_start_factor);
            print('warm_start: %s' % self.warm_start);
            print('ws_factor:  %g' % self.warm_start_factor);
            print('reference:  %s' % ref_string);
            print('lambda:     %g' % self.la);
            print('max_itr:    %g' % self.max_itr);
            print('warm_start: %s' % self.warm_start);
            print('ws_factor:  %g' % self.warm_start_factor);
            print('------------------------------------');


        r_is_empty = not self.r; #  isempty(r)

        if (r_is_empty):
            ps_factor = self.perp_start_factor;
            if (self.perp_start == 'rand'):
                rr = ps_factor*np.random.rand(*batch.shape).astype('float32');
            elif(self.perp_start == 'randn'):
                rr = ps_factor*np.random.randn(*batch.shape).astype('float32');
            elif(self.perp_start == 'ones'):
                rr = ps_factor*np.ones(batch.shape).astype('float32');
            else: # "off"
                rr = ps_factor*np.zeros(batch.shape).astype('float32');
            self.r.append(rr);

        v_is_empty = not self.v; #  isempty(v)

        if (v_is_empty):
            vv = np.zeros(batch.shape).astype('float32');
            self.v.append(vv);

        ws_factor  = self.warm_start_factor;
        warm_start = self.warm_start;
        if (warm_start == 'rand'):
            label = ws_factor*np.random.rand(*batch.shape).astype('float32');
        elif (warm_start == 'randn'):
            label = ws_factor*np.random.randn(*batch.shape).astype('float32');
        elif (warm_start == 'ones'):
            label = ws_factor*np.ones(batch.shape).astype('float32');
        else: # 'off'
            label = np.zeros(batch.shape).astype('float32');    
        
        
        reference = self.reference;
        
        if type(reference) is np.ndarray:
            ref = reference;
        elif type(reference) == str: 
            if reference.lower() == 'pred':
                ref = f(batch);
            elif reference.lower() == 'true':
                ref = batch;
            else:
                raise ValueError("Reference must be \'pred\', \'true\' or a Numpy array");
        else:
            raise ValueError("Reference must be \'pred\', \'true\' or a Numpy array");
        dQ1 = lambda im, r: dQ(im, r, ref+label, self.la);

        
        if self.optimizer == 'SGA':
            rr, vv, backlog = pertub_SGA(f, dQ1, batch, self.max_itr, 
                                         self.r[-1], 
                                         self.v[-1], 
                                         momentum=self.momentum, 
                                         lr=self.learning_rate, 
                                         verbose=self.verbose,
                                         max_r_norm=self.max_r_norm, 
                                         max_diff_norm=self.max_diff_norm);

            self.r.append(rr);
            self.v.append(vv);
            self.backlog = backlog;
        self.x0 = [batch];
