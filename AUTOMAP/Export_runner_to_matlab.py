"""
This file takes the runner object specified by the `runner_id` and store the 
data and perturbations in a Matlab friendly format.
"""


import scipy.io;
import h5py
from os.path import join;
import numpy as np;
from automap_config import src_weights, src_mri_data;
from automap_tools import *;
from Runner import Runner;
from Automap_Runner import Automap_Runner;


runner_id = 52;
runner = load_runner(runner_id);

mri_data = runner.x0[0];
rr = runner.r;

rr[0] = np.zeros(rr[0].shape, dtype=rr[0].dtype);

rr = np.asarray(rr);
print(rr.shape);

fname = "runner_%d_data.mat" % (runner_id);
scipy.io.savemat(join(src_mri_data, fname), {'mri_data': mri_data, 'rr': rr});


