import sys;
import os.path
from os.path import join
import numpy as np;
import matplotlib.pyplot as plt;
from Deep_MRI_Runner import Deep_MRI_Runner;
from deep_mri_tools import load_runner;
from cascadenet.util.helpers import from_lasagne_format, to_lasagne_format;

runner_id = 24;
im_nbr = 1;
runner = load_runner(runner_id);
dest = 'plots_perturbations';

if not os.path.isdir(dest):
    os.mkdir(dest);


print(
"""
Runner: %d

lambda:        %g
momentum:      %g
learning_rate: %g
ps:            %s
ps_factor:     %g
ws:            %s
ws_factor:     %g
max_itr:       %g
""" % (runner_id, runner.la, runner.momentum, 
       runner.learning_rate, 
       runner.perp_start, runner.perp_start_factor,
       runner.warm_start, runner.warm_start_factor,
       runner.max_itr));

max_itr = runner.max_itr;

pert_list = [];
print(len(runner.r))
for i in range(len(runner.r)):
    pert = from_lasagne_format(runner.r[i]);
    pert_im = pert[im_nbr];
    print('i: %d, norm: %g' % (i, np.linalg.norm(pert_im, 'fro')))
    pert_list.append(pert_im);
relevant_perturbations = [0,4,8,12];
N = pert_im.shape[0];

X = np.zeros([len(relevant_perturbations)] + list(pert_im.shape), 
             dtype=pert_im.dtype);


for i in range(len(relevant_perturbations)):
    X[i, :,:] = pert_list[relevant_perturbations[i]];

vmax = np.amax(abs(X));
vmin = np.amin(abs(X));
print('vmin: ', vmin);
print('vmax: ', vmax);
for i in range(len(relevant_perturbations)):
    idx = relevant_perturbations[i];
    fname = 'pert_r_%d_idx_%04d.png' % (runner_id, idx*runner.max_itr);
    fname = join(dest,fname);
    print("""
itr: %g
vmin: %g 
vmax: %g
norm: %g
    """ % (idx*runner.max_itr, 
           np.amin(abs(pert_list[idx])), 
           np.amax(abs(pert_list[idx])),
           np.linalg.norm(abs(pert_list[idx]), 'fro')));
    plt.imsave(fname, abs(pert_list[idx]), vmin=vmin, vmax=vmax, cmap='gray');













