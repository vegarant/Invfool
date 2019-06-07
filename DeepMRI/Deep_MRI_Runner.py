from __future__ import print_function;

from adversarial_tools import l2_norm_of_tensor
from Runner import Runner;
from cascadenet.util.helpers import from_lasagne_format, to_lasagne_format;
from os.path import join;
import numpy as np;
import scipy;

from deep_mri_config import deep_mri_runner_path;

class Deep_MRI_Runner(Runner):
    """ An extension of the Runner class with some specializations to the Deep MRI Net.
    """

    def find_adversarial_perturbation(self, f, dQ, batch, only_real=False):
        """ Search for adversarial perturbation.
        
        An extension of Runner.find_adversarial_perturbation(...) with the 
        additional parameter ``only_real``, which makes the algorithm only 
        search for real adversarial perturbations. 
    
        :param only_real: Search only for real perturbations.

        """
        if only_real:
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

                rr = to_lasagne_format(np.real(from_lasagne_format(rr)));
                self.r.append(rr);

        Runner.find_adversarial_perturbation(self, f, dQ, batch);

    def save_runner(self, f, network_name='DeepMRI', 
                    data_path=deep_mri_runner_path):
        """
        Save the current runner object.

        Reads the current runner count and store the runner object at 
        data_path/data/ as 'runner_count.npz'. In addition the file 
        'runner_description.txt' (located at data_path) is updated with
        statistics from the run. 

        :param f: Neural Network.
        :param network_name: Network Name (str).
        :param data_path: Where to store the runner_object. 
                          The data_path directory should contain a subdirectory 
                          called 'data', where the object is stored.

        :returns: Runner count (int).
        """

        count = self._read_count(data_path);
        fname = join('data', 'runner_%d.npz' % (count));
        fname = join(data_path, fname);
        #print("len(r): ", len(self.r))
        np.savez(fname, x0=self.x0[0], r=self.r, v=self.v, mask=self.mask[0],
                 la=self.la,
                 lr=self.learning_rate, 
                 momentum=self.momentum,
                 smoothing_eps=self.smoothing_eps, 
                 ws=self.warm_start, 
                 wsf=self.warm_start_factor, 
                 ps=self.perp_start, 
                 psf=self.perp_start_factor, 
                 max_itr=self.max_itr, 
                 max_r_norm=self.max_r_norm, 
                 max_diff_norm=self.max_diff_norm,
                 optimizer=self.optimizer,
                 backlog=self.backlog);


        x0   = self.x0[0];
        r    = self.r[-1];
        mask = self.mask[0];
        batch_size = x0.shape[0]; 

        # Evaluate the networks
        fx  = f(x0);
        fxr = f(x0+r);

        x_i1   = from_lasagne_format(x0);
        r_i1   = from_lasagne_format(r);
        fx_i1  = from_lasagne_format(fx);
        fxr_i1 = from_lasagne_format(fxr);

        fname = join(data_path, "runner_description.txt");
        outfile = open(fname, 'a');

        outfile.write("""
-------------------------- %03d: %s --------------------------
        opt: %s, mom: %g, learn_rate: %g, smooth_esp: %g, 
        la: %g, max_r_norm: %g, max_diff_norm: %g,
        max_itr: %d
        ws: %s, wsf: %g, ps: %s psf: %g
----                                                         ----
""" % (
count, network_name, self.optimizer, self.momentum, self.learning_rate,
self.smoothing_eps, self.la, self.max_r_norm, self.max_diff_norm, self.max_itr,
self.warm_start, self.warm_start_factor, self.perp_start, self.perp_start_factor ));        

        for im_nbr in range(batch_size):
            x_i = x_i1[im_nbr];
            r_i = r_i1[im_nbr];
            fx_i = fx_i1[im_nbr];
            fxr_i = fxr_i1[im_nbr];

            nx = l2_norm_of_tensor(x_i);
            nr = l2_norm_of_tensor(r_i);
            n_diff = l2_norm_of_tensor(fx_i - fxr_i);

            next_str = '%02d: |f(x)-f(x+r)|: %g, |r|: %g, |f(x)-f(x+r)|/|r|: %g |r|/|x|: %g\n' % \
                   (im_nbr, abs(n_diff), nr, abs(n_diff)/nr, nr/nx );
            outfile.write(next_str);
        outfile.close();

        return count;

    def export_data(self, fname, r_idx):
        """
        Save the images and perturbations as .mat files.
        """
        r  = self.r[r_idx];
        x0 = self.x0[0];
        mask = self.mask[0];

        r  = from_lasagne_format(r); 
        x0 = from_lasagne_format(x0); 

        scipy.io.savemat(fname+'x0.mat', mdict={'x0': x0});
        scipy.io.savemat(fname+'r.mat',  mdict={'r':  r });
        scipy.io.savemat(fname+'mask.mat',  mdict={'mask':  mask });



