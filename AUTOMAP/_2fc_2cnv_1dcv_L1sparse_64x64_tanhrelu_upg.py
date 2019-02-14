"""
This file was obtained via private communcation with Bo Zhu so that we could
reproduce the network in the paper 'Image reconstruction by domain-transform
manifold learning', Nature 2018.
"""

import scipy.io as sio
import tensorflow as tf
import numpy as np
import math
import time
import h5py

#from libs.activations import lrelu
#from libs.utils import corrupt

from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.contrib.framework.python.ops import variables
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.layers.python.layers import utils

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import standard_ops
from tensorflow.python.ops import variable_scope

from collections import namedtuple

slim = tf.contrib.slim

# %% Autoencoder definition
def network(batch_size,precision,resolution,in_dim,h_dim, out_dim,model_in_vars=None, model_in_shapes=None, trainable_model_in='True'):

    if precision == 'FP32':
        prec = tf.float32
    elif precision == 'FP16':
        prec = tf.float16

    #dropout_keep_prob = 0.7

    # <editor-fold desc="Definitions">
    x = tf.placeholder(prec, [None, in_dim], name='x')

    y_target = tf.placeholder(prec, [None, out_dim], name='y_target')
    target_output = y_target


    x = tf.cast(x, prec)

    # xmean, xvar = tf.nn.moments(x, [1])
    # x = tf.div(x, tf.expand_dims(tf.sqrt(xvar), 1))

    y_target = tf.cast(y_target, tf.float32)


    if trainable_model_in == 'True':
        trainable_var = True
    elif trainable_model_in == 'False':
        trainable_var = False



    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    sparseflag = tf.placeholder(tf.int32, name='sparseflag')
    sparseflag = tf.cast(sparseflag,tf.int32)


    # </editor-fold>
    # <editor-fold desc="Corrupt Input">
    corrupt_prob = tf.placeholder(prec,[1])

    x_noise = tf.multiply(x, tf.random_uniform(shape=tf.shape(x),
                                                minval=0.99,
                                                maxval=1.01, dtype=prec
                                                )) * corrupt_prob + x * (1 - corrupt_prob)

    #x_noise = tf.mul(x,tf.convert_to_tensor(np.random.binomial(1,1-corrupt_prob,[batch_size, in_dim]),dtype=tf.float32))

    # x_mask = tf.cast(tf.multinomial(tf.mul(([[tf.log(corrupt_prob[0]), tf.log(1-corrupt_prob[0])]]), tf.cast(tf.ones([batch_size, 1]),prec)), in_dim),prec)

    #x_noise = tf.mul(x_mask, x)

    # x_noise = x

    # </editor-fold>
    # <editor-fold desc="Placeholder Bookkeeping">
    W1_fc_plh = tf.placeholder(prec, None)
    W2_fc_plh = tf.placeholder(prec, None)
    W3_fc_plh = tf.placeholder(prec, None)

    b1_fc_plh = tf.placeholder(prec, None)
    b2_fc_plh = tf.placeholder(prec, None)
    b3_fc_plh = tf.placeholder(prec, None)

    W1_cnv_plh = tf.placeholder(tf.float32, None)
    W2_cnv_plh = tf.placeholder(tf.float32, None)
    W3_cnv_plh = tf.placeholder(tf.float32, None)

    b1_cnv_plh = tf.placeholder(tf.float32, None)
    b2_cnv_plh = tf.placeholder(tf.float32, None)
    b3_cnv_plh = tf.placeholder(tf.float32, None)

    W1_dcv_plh = tf.placeholder(tf.float32, None)
    W2_dcv_plh = tf.placeholder(tf.float32, None)
    W3_dcv_plh = tf.placeholder(tf.float32, None)

    b1_dcv_plh = tf.placeholder(tf.float32, None)
    b2_dcv_plh = tf.placeholder(tf.float32, None)
    b3_dcv_plh = tf.placeholder(tf.float32, None)


    if model_in_vars != None:
        if 'W1_fc' in model_in_vars:
            ind = model_in_vars.index('W1_fc')
            W1_fc_plh = tf.placeholder(prec, shape=model_in_shapes[ind])
        if 'W2_fc' in model_in_vars:
            ind = model_in_vars.index('W2_fc')
            W2_fc_plh = tf.placeholder(prec, shape=model_in_shapes[ind])
        if 'W3_fc' in model_in_vars:
            ind = model_in_vars.index('W3_fc')
            W3_fc_plh = tf.placeholder(prec, shape=model_in_shapes[ind])
        if 'b1_fc' in model_in_vars:
            ind = model_in_vars.index('b1_fc')
            b1_fc_plh = tf.placeholder(prec, shape=model_in_shapes[ind])
        if 'b2_fc' in model_in_vars:
            ind = model_in_vars.index('b2_fc')
            b2_fc_plh = tf.placeholder(prec, shape=model_in_shapes[ind])
        if 'b3_fc' in model_in_vars:
            ind = model_in_vars.index('b3_fc')
            b3_fc_plh = tf.placeholder(prec, shape=model_in_shapes[ind])
        if 'W1_cnv' in model_in_vars:
            ind = model_in_vars.index('W1_cnv')
            W1_cnv_plh = tf.placeholder(tf.float32, shape=model_in_shapes[ind])
        if 'W2_cnv' in model_in_vars:
            ind = model_in_vars.index('W2_cnv')
            W2_cnv_plh = tf.placeholder(tf.float32, shape=model_in_shapes[ind])
        if 'W3_cnv' in model_in_vars:
            ind = model_in_vars.index('W3_cnv')
            W3_cnv_plh = tf.placeholder(tf.float32, shape=model_in_shapes[ind])
        if 'b1_cnv' in model_in_vars:
            ind = model_in_vars.index('b1_cnv')
            b1_cnv_plh = tf.placeholder(tf.float32, shape=model_in_shapes[ind])
        if 'b2_cnv' in model_in_vars:
            ind = model_in_vars.index('b2_cnv')
            b2_cnv_plh = tf.placeholder(tf.float32, shape=model_in_shapes[ind])
        if 'b3_cnv' in model_in_vars:
            ind = model_in_vars.index('b3_cnv')
            b3_cnv_plh = tf.placeholder(tf.float32, shape=model_in_shapes[ind])
        if 'W1_dcv' in model_in_vars:
            ind = model_in_vars.index('W1_dcv')
            W1_dcv_plh = tf.placeholder(tf.float32, shape=model_in_shapes[ind])
        if 'W2_dcv' in model_in_vars:
            ind = model_in_vars.index('W2_dcv')
            W2_dcv_plh = tf.placeholder(tf.float32, shape=model_in_shapes[ind])
        if 'W3_dcv' in model_in_vars:
            ind = model_in_vars.index('W3_dcv')
            W3_dcv_plh = tf.placeholder(tf.float32, shape=model_in_shapes[ind])
        if 'b1_dcv' in model_in_vars:
            ind = model_in_vars.index('b1_dcv')
            b1_dcv_plh = tf.placeholder(tf.float32, shape=model_in_shapes[ind])
        if 'b2_dcv' in model_in_vars:
            ind = model_in_vars.index('b2_dcv')
            b2_dcv_plh = tf.placeholder(tf.float32, shape=model_in_shapes[ind])
        if 'b3_dcv' in model_in_vars:
            ind = model_in_vars.index('b3_dcv')
            b3_dcv_plh = tf.placeholder(tf.float32, shape=model_in_shapes[ind])


    print('test')
    # </editor-fold>

    pad_size = 4

    num_filters_1 = 64
    kernel_size_1 = 5
    stride_size_1 = 1

    num_filters_2 = 64
    kernel_size_2 = 5
    stride_size_2 = 1

    num_filters_3 = 64
    kernel_size_3 = 5
    stride_size_3 = 1

    num_filters_4 = 64
    kernel_size_4 = 7
    stride_size_4 = 1

    fast_train_vars = []
    slow_train_vars = []

    # ==== 1FC LAYER

    current_input = x_noise
    n_input = in_dim
    n_output = h_dim

    with tf.device("/cpu:0"):

        # output = slim.dropout(output, keep_prob)

        if 'W1_fc' in model_in_vars:
            W1_fc = tf.Variable(W1_fc_plh, name='W1_fc')
            b1_fc = tf.Variable(b1_fc_plh, name='b1_fc')

            slow_train_vars.append(W1_fc)
            slow_train_vars.append(b1_fc)


        else:
            W1_fc = tf.Variable(tf.random_uniform([n_input, n_output],
                                                  -1.0 / math.sqrt(n_input),
                                                  1.0 / math.sqrt(n_input), dtype=prec), dtype=prec, name='W1_fc')

            b1_fc = tf.Variable(tf.zeros([n_output], dtype=prec), dtype=prec, name='b1_fc')

            fast_train_vars.append(W1_fc)
            fast_train_vars.append(b1_fc)

        output = tf.nn.tanh(tf.matmul(current_input, W1_fc) + b1_fc)
        output_1fc = output
        # output = slim.dropout(output, keep_prob)

    # ====2 FC LAYER

    current_input = output

    n_input = n_output
    n_output = out_dim

    with tf.device("/cpu:0"):

        if 'W2_fc' in model_in_vars:

            W2_fc = tf.Variable(W2_fc_plh, name='W2_fc')
            b2_fc = tf.Variable(b2_fc_plh, name='b2_fc')

            slow_train_vars.append(W2_fc)
            slow_train_vars.append(b2_fc)

        else:
            W2_fc = tf.Variable(
                tf.random_uniform([n_input, n_output],
                                  -1.0 / math.sqrt(n_input),
                                  1.0 / math.sqrt(n_input), dtype=prec), dtype=prec, name='W2_fc')
            b2_fc = tf.Variable(tf.zeros([n_output], dtype=prec), dtype=prec, name='b2_fc')

            fast_train_vars.append(W2_fc)
            fast_train_vars.append(b2_fc)

        # output = tf.nn.tanh(tf.matmul(current_input, W2_fc) + b2_fc)
        output = tf.matmul(current_input, W2_fc) + b2_fc
        preconv = output


        x_noise = output

        # ZERO MEAN THE OUTPUT

        x_noise = x_noise - tf.expand_dims(tf.reduce_mean(x_noise, 1), 1)

        x_noise = tf.cast(x_noise, tf.float32)

        # RESHAPE

        if len(x.get_shape()) == 2:
            x_dim = np.sqrt(x_noise.get_shape().as_list()[1])
            print('===========XNOISE SHAPE')
            print(x_noise.get_shape())
            if x_dim != int(x_dim):
                raise ValueError('Unsupported input dimensions')
            x_dim = int(x_dim)

            # x_complex = tf.complex( x_noise[:,:x_dim*x_dim] , x_noise[:, x_dim*x_dim:])

            # x_mag = tf.abs(x_complex)
            x_mag = x_noise

            print(x_mag.get_shape())
            print(x_dim)

            x_tensor = tf.reshape(
                x_mag, [batch_size, x_dim, x_dim, 1])
            x_tensor = tf.pad(x_tensor, [[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]],
                              "SYMMETRIC")
        elif len(x_noise.get_shape()) == 4:
            x_tensor = x_noise
        else:
            raise ValueError('Unsupported input dimensions')


        print(x_tensor.get_shape())


        def get_deconv_dim(dim_size, stride_size, kernel_size, padding):
            if isinstance(dim_size, ops.Tensor):
                dim_size = math_ops.mul(dim_size, stride_size)
            elif dim_size is not None:
                dim_size *= stride_size

            if padding == 'VALID' and dim_size is not None:
                dim_size += max(kernel_size - stride_size, 0)
            return dim_size


        #==== 1CNV LAYER

        input_1_cnv = x_tensor

        weights_shape = [kernel_size_1, kernel_size_1, 1, num_filters_1]

        if 'W1_cnv' in model_in_vars:
            W1_cnv = tf.Variable(W1_cnv_plh, name='W1_cnv')
            b1_cnv = tf.Variable(b1_cnv_plh, name='b1_cnv')
            print('LOADED 1st LAYER')
            fast_train_vars.append(W1_cnv)
            fast_train_vars.append(b1_cnv)
        else:
            W1_cnv = variables.model_variable('W1_cnv', initializer=initializers.xavier_initializer(),
                                                     shape=weights_shape, dtype=tf.float32)
            b1_cnv = variables.model_variable('b1_cnv', initializer=init_ops.zeros_initializer, shape=[num_filters_1, ], dtype=tf.float32)
            fast_train_vars.append(W1_cnv)
            fast_train_vars.append(b1_cnv)

        net = tf.nn.conv2d(input_1_cnv, W1_cnv, strides=[1, stride_size_1, stride_size_1, 1], padding='SAME')
        net = (tf.add(net, b1_cnv))
        net = tf.tanh(net)
        print('====1 CNV SHAPE =====')
        print(net.get_shape())

        cnv1_out = net

        #net = slim.dropout(net, keep_prob, is_training=trainable_model_in)

        # ==== 2CNV LAYER

        input_2_cnv = net

        weights_shape = [kernel_size_2, kernel_size_2, num_filters_1, num_filters_2]

        if 'W2_cnv' in model_in_vars:
            W2_cnv = tf.Variable(W2_cnv_plh, name='W2_cnv')
            b2_cnv = tf.Variable(b2_cnv_plh, name='b2_cnv')
            print('LOADED 1st LAYER')
            fast_train_vars.append(W2_cnv)
            fast_train_vars.append(b2_cnv)
        else:
            W2_cnv = variables.model_variable('W2_cnv', initializer=initializers.xavier_initializer(),
                                              shape=weights_shape, dtype=tf.float32)
            b2_cnv = variables.model_variable('b2_cnv', initializer=init_ops.zeros_initializer, shape=[num_filters_2, ],
                                              dtype=tf.float32)
            fast_train_vars.append(W2_cnv)
            fast_train_vars.append(b2_cnv)

        net = tf.nn.conv2d(input_2_cnv, W2_cnv, strides=[1, stride_size_2, stride_size_2, 1], padding='SAME')
        net = tf.nn.relu(tf.add(net, b2_cnv))
        print('====2 CNV SHAPE =====')
        print(net.get_shape())

        cnv2_out = net
        # net = slim.dropout(net, keep_prob, is_training=trainable_model_in)

    # with tf.device("/gpu:0"):
    #     # ==== 3CNV LAYER
    #
    #     input_3_cnv = net
    #
    #     weights_shape = [kernel_size_3, kernel_size_3, num_filters_2, num_filters_3]
    #
    #     if 'W3_cnv' in model_in_vars:
    #         W3_cnv = tf.Variable(W3_cnv_plh, name='W3_cnv')
    #         b3_cnv = tf.Variable(b3_cnv_plh, name='b3_cnv')
    #         print('LOADED 1st LAYER')
    #
    #         fast_train_vars.append(W3_cnv)
    #         fast_train_vars.append(b3_cnv)
    #     else:
    #         W3_cnv = variables.model_variable('W3_cnv', initializer=initializers.xavier_initializer(),
    #                                           shape=weights_shape, dtype=tf.float32)
    #         b3_cnv = variables.model_variable('b3_cnv', initializer=init_ops.zeros_initializer, shape=[num_filters_3, ],
    #                                           dtype=tf.float32)
    #         fast_train_vars.append(W3_cnv)
    #         fast_train_vars.append(b3_cnv)
    #
    #     net = tf.nn.conv2d(input_3_cnv, W3_cnv, strides=[1, stride_size_3, stride_size_3, 1], padding='SAME')
    #     net = tf.nn.relu(tf.add(net, b3_cnv))
    #     print('====3 CNV SHAPE =====')
    #     print(net.get_shape())
    #     cnv3_out = net

        featx = net.get_shape().as_list()[1]
        featy = net.get_shape().as_list()[2]

        codes = cnv2_out
        cnv2_size = batch_size*featx*featy*num_filters_2

        #=== SPARSIFY

        # # TOP 1
        # print('max net')
        # maxarray = tf.reduce_max(net,[0,1,2])
        # featx = net.get_shape().as_list()[1]
        # featy = net.get_shape().as_list()[2]
        #
        #
        # # maskarray = tf.tile(tf.expand_dims(tf.expand_dims(maxarray, 1), 1), [1,featx,featy, 1])*0.5
        # # binarymask = tf.cast(tf.greater_equal(net,maskarray),tf.float32)
        # # sparsearray = tf.mul(net,binarymask)
        #
        # #maxarray = tf.reduce_max(maxarray,[0])
        # print('max net')
        # print(maxarray.get_shape())
        # maskarray = tf.tile((tf.expand_dims(tf.expand_dims(tf.expand_dims(maxarray, 0), 0),0)), [batch_size, featx, featy, 1]) * 0.2
        # binarymask = tf.cast(tf.greater_equal(net, maskarray), tf.float32)
        # sparsearray = tf.mul(net, binarymask)


        # print(tf.reduce_max(net,[1,2]).get_shape())
        # print(featx)
        # print(tf.tile(tf.expand_dims(tf.expand_dims(maxarray,1),1),[1,featx,featy,1]).get_shape())



        # TOP K

        # featx = net.get_shape().as_list()[1]
        # featy = net.get_shape().as_list()[2]
        # numfilt = net.get_shape().as_list()[3]
        # swappednet = tf.transpose(tf.reshape(net,[batch_size,featx*featy,numfilt]),[0,2,1])
        # topkarray = tf.nn.top_k(swappednet,k=tf.cast(tf.round(0.01*featx*featy),tf.int32),sorted=False)
        #
        # print('topkvalues shape')
        # print(topkarray.values[:,:,-1].get_shape())
        #
        # topkvalarray = topkarray.values[:,:,-1]
        # maskarray = tf.tile(tf.expand_dims(tf.expand_dims(topkvalarray, 1), 1), [1, featx, featy, 1]) - 0.0000001
        # sparsearray = tf.mul(tf.maximum(net, maskarray), net)


        print('===SPARSEFLAG====')
        print(sparseflag)

        #==== 1DCV LAYER

        num_filters_out = 1

        # def f1():
        #     return sparsearray
        #
        # def f2():
        #     return net
        #
        # r = tf.cond(tf.equal(sparseflag, 1), f1, f2)

        input_deconv = net

        # input_deconv = sparsearray

        # if sparseflag[0] == 1:
        #     input_deconv = sparsearray
        # elif sparseflag[0] == 0:
        #     input_deconv = net
        # else:
        #     input_deconv = net

        # W1_dcv = W1_cnv

        weights_shape = [kernel_size_4, kernel_size_4, 1, num_filters_4]

        if 'b1_dcv' in model_in_vars:
            b1_dcv = tf.Variable(b1_dcv_plh, name='b1_dcv', trainable=trainable_var)
            print('=================================LOADED B1_DCV')
            if trainable_var == True:
                 fast_train_vars.append(b1_dcv)
        else:
            print('=================================B1_DCV NOT IN MODEL')
            b1_dcv = variables.model_variable('b1_dcv', initializer=init_ops.zeros_initializer, shape=[num_filters_out, ],
                                              dtype=tf.float32)
            fast_train_vars.append(b1_dcv)

        if 'W1_dcv' in model_in_vars:
            W1_dcv = tf.Variable(W1_dcv_plh, name='W1_dcv', trainable=trainable_var)
            if trainable_var == True:
                 fast_train_vars.append(W1_dcv)
        else:
            W1_dcv = variables.model_variable('W1_dcv', initializer=initializers.xavier_initializer(),
                                              shape=weights_shape, dtype=tf.float32)
            fast_train_vars.append(W1_dcv)

        deconv = tf.nn.conv2d_transpose(input_deconv, W1_dcv, tf.stack(
            [input_1_cnv.get_shape()[0], input_1_cnv.get_shape()[1], input_1_cnv.get_shape()[2],
             input_1_cnv.get_shape()[3]]),
                                        strides=[1, stride_size_1, stride_size_1, 1], padding='SAME')

        # out_shape = input_deconv.get_shape().as_list()
        # out_shape[-1] = num_filters_out
        # out_shape[1] = get_deconv_dim(out_shape[1], stride_size_1, kernel_size_1, 'SAME')
        # out_shape[2] = get_deconv_dim(out_shape[2], stride_size_1, kernel_size_1, 'SAME')
        # # deconv.set_shape(out_shape)

        # net = deconv

        net = tf.add(deconv, b1_dcv)

        # net = (tf.add(net, input_1_cnv))

        y = net

        y_target_2d = tf.reshape(
            y_target, [batch_size, x_dim, x_dim, 1])
        y_target_2d = tf.pad(y_target_2d, [[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]],
                          "SYMMETRIC")


        ycrop = y[0:batch_size, pad_size:pad_size + x_dim, pad_size:pad_size + x_dim, 0]
        xt_crop = x_tensor[0:batch_size, pad_size:pad_size + x_dim, pad_size:pad_size + x_dim, 0]
        yt_crop = y_target_2d[0:batch_size, pad_size:pad_size + x_dim, pad_size:pad_size + x_dim, 0]

        ycrop_vec = tf.reshape(ycrop, [batch_size, x_dim * x_dim])
        xtcrop_vec = tf.reshape(xt_crop, [batch_size, x_dim * x_dim])
        ytcrop_vec = tf.reshape(yt_crop, [batch_size, x_dim * x_dim])

        ycrop_0mean = ycrop_vec - tf.expand_dims(tf.reduce_mean(ycrop_vec, 1), 1)
        xtcrop_0mean = xtcrop_vec - tf.expand_dims(tf.reduce_mean(xtcrop_vec, 1), 1)
        ytcrop_0mean = ytcrop_vec - tf.expand_dims(tf.reduce_mean(ytcrop_vec, 1), 1)

        ycmean, ycvar = tf.nn.moments(ycrop_0mean, [1])
        ycnorm = tf.div(ycrop_0mean, tf.expand_dims(tf.sqrt(ycvar), 1))

        ytcmean, ytcvar = tf.nn.moments(ytcrop_0mean, [1])
        ytcnorm = tf.div(ytcrop_0mean, tf.expand_dims(tf.sqrt(ytcvar), 1))

        # cost = tf.reduce_sum(tf.to_float(tf.square(ycrop_0mean - ytcrop_0mean))) + 0.1*tf.reduce_sum(tf.sqrt(tf.square(cnv3_out)+0.00001))/(cnv3_size)

        nonzeros = tf.cast(tf.greater(tf.abs(cnv2_out), 0), tf.float32)
        nonzero_frac = tf.reduce_sum(nonzeros)/cnv2_size

        l1_penalty = 0.0001 * tf.reduce_sum(tf.abs(cnv2_out)) # (tf.reduce_sum(tf.abs(cnv1_out)) + tf.reduce_sum(tf.abs(cnv2_out)) + tf.reduce_sum(tf.abs(cnv3_out)))



        cost = tf.reduce_sum(tf.to_float(tf.square(ycrop_0mean - ytcrop_0mean)))  + l1_penalty#  * 10*tf.square(nonzero_frac-0.10)

        # cost = tf.reduce_sum(tf.to_float(tf.square(x_noise - y_target)))




        # debug = tf.reduce_sum(input_deconv)

        debug = [nonzero_frac, l1_penalty / (batch_size * x_dim * x_dim)]

        # debug = tf.to_float(binarymask.get_shape().as_list())
        #debug = maskarray[:,0,0,:]

        valcost = tf.reduce_sum(tf.to_float(tf.square(ycrop_0mean - ytcrop_0mean)))

        #ebug = sparseflag
        # %% cost function measures pixel-wise difference

        #cost = tf.reduce_sum(tf.to_float(tf.square(y - x_tensor)))
        # valcost = cost
        tf.summary.scalar('cost', cost)

    print('====TRAINABLE VARIABLES')
    paramtuple_list = []
    param_data_list = []
    for var in range(len(tf.trainable_variables())):
        #print(tf.trainable_variables()[var].name.split(':')[0])
        #print(tf.trainable_variables()[var])
        paramtuple_list.append(tf.trainable_variables()[var].name.split(':')[0])
        param_data_list.append(tf.trainable_variables()[var])

    params = namedtuple('params', paramtuple_list)

    Modelout = params._make(param_data_list)
    #print(Modelout)

    # VA: the next line is added by Vegard Antun so that the image does not
    # change its' rotation.
    ycrop = tf.transpose(ycrop, perm=[0,2,1]);
    return {'x': x, 'y': y, 'ycrop': ycrop, 'cost': cost,'valcost': valcost,'y_target':y_target,'keep_prob':keep_prob,'corrupt_prob': corrupt_prob,
            'Modelout':Modelout, 'debug':debug,'Fast_train_vars':fast_train_vars,'Slow_train_vars':slow_train_vars,'sparseflag':sparseflag,
            'W1_fc':W1_fc_plh,'W2_fc':W2_fc_plh,'W3_fc':W3_fc_plh,'output_1fc':output_1fc,'preconv':preconv,'codes':codes,
            'b1_fc':b1_fc_plh,'b2_fc':b2_fc_plh,'b3_fc':b3_fc_plh,
            'W1_cnv': W1_cnv_plh, 'W2_cnv': W2_cnv_plh, 'W3_cnv': W3_cnv_plh,
            'b1_cnv': b1_cnv_plh, 'b2_cnv': b2_cnv_plh, 'b3_cnv': b3_cnv_plh,
            'W1_dcv': W1_dcv_plh, 'W2_dcv': W2_dcv_plh, 'W3_dcv': W3_dcv_plh,
            'b1_dcv': b1_dcv_plh, 'b2_dcv': b2_dcv_plh, 'b3_dcv': b3_dcv_plh  }
