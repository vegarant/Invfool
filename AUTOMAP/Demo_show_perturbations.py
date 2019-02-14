"""
This script loads a runner object an writes the perturbations it image files.
"""

import sys;
import os.path
from os.path import join
import numpy as np;
from Automap_Runner import Automap_Runner;
from automap_tools import load_runner;
import matplotlib.pyplot as plt;

runner_id1 = 50;
runner_id2 = 52;
runner1 = load_runner(runner_id1);
runner2 = load_runner(runner_id2);
dest = 'plots_perturbations';

if not os.path.isdir(dest):
    os.mkdir(dest);


print(runner1)
print(runner2)

max_itr = runner1.max_itr;
im_nbr = [3, 4, 5];
relevant_pert1 = [(3,1), (3,2), (4,3), (5,4)]; # (im_nbr, idx)
relevant_pert2 = [(3,1), (3,2), (3,3), (5,4)]; # (im_nbr, idx)


pert_list = [];
N = 128;

num_elements = len(relevant_pert1)+len(relevant_pert2);
X = np.zeros([num_elements, N, N], 
             dtype=runner1.x0[0].dtype);

for i in range(len(relevant_pert1)):
    im_nbr1, idx = relevant_pert1[i];
    X[i,:,:] = abs(runner1.r[idx][im_nbr1, :, :]);

for i in range(len(relevant_pert2)):
    im_nbr1, idx = relevant_pert2[i];
    X[len(relevant_pert1)+i,:,:] = abs(runner2.r[idx][im_nbr1, :, :]);

vmax = np.amax(X);
vmin = np.amin(X);
print('vmin: ', vmin);
print('vmax: ', vmax);

itr1 = [12,16,20,24]
itr2 = [160, 170, 177, 183];

for i in range(len(relevant_pert1)):
    im_nbr1, idx = relevant_pert1[i];
    fname = 'pert_r_%d_p_im_nbr_%d_itr_%02d.png' % (runner_id1, im_nbr1, itr1[i]);
    fname = join(dest,fname);
    print("""
itr:  %g
vmin: %g 
vmax: %g
norm: %g
    """ % (itr1[i], 
           np.amin(X[i,:,:]), 
           np.amax(X[i,:,:]),
           np.linalg.norm(X[i,:,:], 'fro')));
    plt.imsave(fname, X[i,:,:], vmin=vmin, vmax=vmax, cmap='gray');

for i in range(len(relevant_pert2)):
    im_nbr1, idx = relevant_pert2[i];
    fname = 'pert_r_%d_p_im_nbr_%d_itr_%02d.png' % (runner_id2, im_nbr1, itr2[i]);
    fname = join(dest,fname);
    print("""
itr:  %g
vmin: %g 
vmax: %g
norm: %g
    """ % (itr2[i], 
           np.amin(X[len(relevant_pert1) + i,:,:]), 
           np.amax(X[len(relevant_pert1) + i,:,:]),
           np.linalg.norm(X[len(relevant_pert1) + i,:,:], 'fro')));
    plt.imsave(fname, X[len(relevant_pert1) + i,:,:], vmin=vmin, vmax=vmax, cmap='gray');

