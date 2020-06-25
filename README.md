# On instabilities of deep learning in image reconstruction - Does AI come at a cost?

This repository contains the code from the paper "On instabilities of deep
learning in image reconstruction - Does AI come at a cost?", by V. Antun, F.
Rennar, C. Poon, B. Adcock and A. Hansen.

In order to make this code run you will have to download and install
the neural networks we have considered. Most of the necessary data can be
downloaded from
[http://folk.uio.no/vegarant/123/storage2.zip](http://folk.uio.no/vegarant/123/storage2.zip)
(4.9 GB). Please note that you will have to modify all paths in the
source files so that they point to the data. You will also need to
add the directory `py_adv_tools` to your python path. 

For the state of the art reconstruction we have used the
[ShearletReweighting](https://github.com/jky-ma/ShearletReweighting)
code from J. Ma & M. März paper and
[spgl1](https://github.com/mpf/spgl1).  These repositories must also
be downloaded and added to your Matlab path.

## FBPConvNet - Ell 50 and Med 50
To test the FBPConvNet you will have to download and install
[MatConvNet](http://www.vlfeat.org/matconvnet/) and the
[FBPConvNet](https://github.com/panakino/FBPConvNet) and add these repositories
on you matlab path. From within the invfool/FBPConvNet directory you should then be
able to run the scripts.  

## Deep MRI Net
Download and install the
[DeepMRINet](https://github.com/js3611/Deep-MRI-Reconstruction) and
add it to your pythonpath. Note that to run DeepMRINet you need a very
specific version of Theano and Lasagne. See the GitHub page of
DeepMRINet for more information about this. Then run the code in
the `DeepMRI` folder.

## MRI-VN
Download the network code for
[MRI-VN](https://github.com/VLOGroup/mri-variationalnetwork) and add it to your
python path. Note that this network requires a custom-made version of
tensorflow, [tensorflow-icg](https://github.com/VLOGroup/tensorflow-icg). To 
run "the add more samples" experiment you need to download the data from 
[GLOBUS](https://app.globus.org/file-manager?origin_id=15c7de28-a76b-11e9-821c-02b7a92d8e58&origin_path=%2F).


## AUTOMAP 
These experiments are self contained. It requires a vanilla
Tensorflow install. 

## DAGAN
The original code for this network can be found at
[this](https://github.com/nebulaV/DAGAN) GitHub page. Full dataset can be
downloaded from MICCAI 2013 grand challenge
[page](https://my.vanderbilt.edu/masi/workshops/). We provide all code and
data necessary to reproduce the figures in the paper. We do not provide the
code to train the network nor the full dataset, as this can be found via
the links above. To run the code you need Tensorflow and Tensorlayer.  

