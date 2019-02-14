from Runner import Runner;
import os;
import pickle 


class Variational_Runner(Runner):
    """ An extension of the Runner class with some specializations to the Variational network 
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
                 optimizer='SGA',
                 momentum = 0.9,
                 smoothing_eps = 1e-8,
                 learning_rate = 0.01,
                 verbose=True,
                 mask=None,
                 coil_sens=None
                 ):
        Runner.__init__(self, 
                 max_itr, 
                 max_r_norm,
                 max_diff_norm,
                 la, 
                 warm_start, # ||f(x+r) - f(x) + p||_{2}^{2} 
                 warm_start_factor,
                 perp_start,
                 perp_start_factor, 
                 optimizer,
                 momentum,
                 smoothing_eps,
                 learning_rate,
                 verbose,
                 mask);
        self.coil_sens = coil_sens;
    
    
    def save_runner(self, data_path):
        count = self._read_count(data_path);
        fname = 'runner_%03d.pkl' % count;
        fname = os.path.join(data_path, 'data', fname);
        fID = open(fname, 'wb');
        pickle.dump(self, fID);
        fID.close();
        return count;
