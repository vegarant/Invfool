import numpy as np;
import os;
import pickle;
import mriutils;

model_dict = {\
'mri_vn_2018-08-06--12-12-15': 'masks/un_samp_patt_0.15.mat'
};

def get_model_name_and_mask_name(samp_type, samp_fraction):
    if samp_type=='rand':
        type1 = '';
    elif samp_type == 'unif':
        type1 = 'un_';
    else:
        print('Unknown type');

    true_mask_name = 'masks/%ssamp_patt_%g.mat' % (type1, samp_fraction);
    for key in model_dict.keys():
        mask_name = model_dict[key];
        if (mask_name == true_mask_name):
            return key, mask_name;

    return None;

def get_uniform_sampling_patt(subsampling_fact, 
                              nbr_center_lines = 28,
                              s = [640, 368]):
    #s = [640, 368];
    s0 = s[0];
    s1 = s[1];
    
    tot_pix = np.prod(s);
    
    nbr_lines = int(np.ceil((tot_pix*subsampling_fact)/s0));
    
    
    nbr_remaning_lines = int(nbr_lines-nbr_center_lines);
    
    
    nbr_lines_left  = int(np.ceil(nbr_remaning_lines/2));
    nbr_lines_right = int(np.floor(nbr_remaning_lines/2));
    
    
    left_up_bound   = int(s1/2-nbr_center_lines/2);
    right_low_bound = int(s1/2+nbr_center_lines/2);
    
    step_left  = int(round(left_up_bound/nbr_lines_left));
    step_right = int(round((s1-right_low_bound)/nbr_lines_right));
    
    
    idx_left  = np.arange(0, left_up_bound, step_left); 
    idx_right = np.arange(right_low_bound+2,s1, step_right);
    idx_center = np.arange(left_up_bound, right_low_bound+1);
    
    idx = np.concatenate([idx_left, idx_center, idx_right]);
    
    out_arr = np.zeros(s, dtype='uint8');
    out_arr[:,idx] = 1

    return out_arr;

def load_runner(runner_id, data_path):
    fname = 'runner_%03d.pkl' % runner_id;
    fname = os.path.join(data_path, 'data', fname);
     
    fID = open(fname, 'rb');
    runner = pickle.load(fID);
    
    fID.close()
    return runner;


def hand_f(val_fn, x, coil_sens, mask):

    # Assumes x.shape         =  (1,640,368);
    # Assumes coil_sens.shape = (1,15, 640, 368);
    # Assumes mask.shape      =  (1, 640, 368);
    coil_sens = np.squeeze(coil_sens);
   
    kspace = mriutils.mriForwardOp(x, coil_sens, mask);
    # Then kspace.shape = (15,640, 368);
    #print('coil_sens.shape: ', coil_sens.shape);
    #print('kspace.shape: ', kspace.shape);
    #print('mask.shape: ', mask.shape);
    x_adj = mriutils.mriAdjointOp(kspace, coil_sens, mask);
    # and 
    # x_adj.shape = (1, 640, 368);
    x_adj = np.asarray([x_adj]);  
    kspace = np.asarray([kspace]);  
    coil_sens = np.asarray([coil_sens]);  
    #print('x_adj.shape: ', x_adj.shape);
    
    #x_adj = Adjoint_Sampling(Sampling(x, mask), mask);

    return val_fn(x_adj, kspace, coil_sens, mask);

def hand_dQ(val_df, x, r, pred, lambda1, coil_sens, mask):
    # Assumes x.shape         =  (1,640,368);
    # Assumes coil_sens.shape = (1,15, 640, 368);
    # Assumes mask.shape      =  (1, 640, 368);
    coil_sens = np.squeeze(coil_sens);

    # Sample image and perturbation.
    kspace_x = mriutils.mriForwardOp(x, coil_sens, mask);
    kspace_r = mriutils.mriForwardOp(r, coil_sens, mask);
    #print('coil_sens.shape: ', coil_sens.shape);
    #print('kspace_x.shape: ',    kspace_x.shape);
    #print('mask.shape: ',      mask.shape);

    x_adj = mriutils.mriAdjointOp(kspace_x, coil_sens, mask);
    r_adj = mriutils.mriAdjointOp(kspace_r, coil_sens, mask);

    x_adj = np.asarray([x_adj]); 
    r_adj = np.asarray([r_adj]); 
    kspace_x = np.asarray([kspace_x]); 
    kspace_r = np.asarray([kspace_r]); 
    coil_sens = np.asarray([coil_sens]); 

    # This is the direct gradient of the network
    df = val_df(x_adj+r_adj, kspace_x + kspace_r, coil_sens, mask, pred);
    #print('df.shape: ', df.shape)
    #print('r.shape: ', r.shape)

    coil_sens = np.squeeze(coil_sens);
    kspace_df = mriutils.mriForwardOp(df, coil_sens, mask);

    #print('kspace_df.shape: ', kspace_df.shape)

    df_adj = mriutils.mriAdjointOp(kspace_df, coil_sens, mask);

    # Performing the last two steps of the backpropagation by hand 

    return df_adj - lambda1*r;
