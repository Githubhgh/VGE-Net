# The following code snippet is inspired and borrowed from the probabilistic_unet project (https://github.com/SimonKohl/probabilistic_unet).
# Author: Kohl, Simon AA
# The following code snippet is inspired and borrowed from the vunet project (https://github.com/CompVis/vunet).
# Author: Patrick Esser

import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.contrib.framework.python.ops import arg_scope



def model_arg_scope(**kwargs):
    counters = {}
    return arg_scope(
        [conv2d, deconv2d, residual_block, activate],
        counters=counters, **kwargs)


def int_shape(x):
    return x.shape.as_list()


def get_name(layer_name, counters):
    ''' utlity for keeping track of layer names '''
    if not layer_name in counters:
        counters[layer_name] = 0
    name = layer_name + '_' + str(counters[layer_name])
    counters[layer_name] += 1
    return name


@add_arg_scope
def conv2d(x, num_filters, filter_size=[3, 3], stride=[1, 1], pad='SAME', init_scale=1., counters={}, init=False, scale=False,
           **kwargs):
    ''' weight normalization convolutional layer '''

    num_filters = int(num_filters)
    strides = [1] + stride + [1]
    name = get_name('conv2d', counters)
    with tf.variable_scope(name):
        xs = x.shape.as_list()
        V = tf.get_variable('V', filter_size + [xs[-1], num_filters],
                            tf.float32, tf.random_normal_initializer(0, 0.05))
        g = tf.get_variable('g', [num_filters], dtype=tf.float32, initializer=tf.constant_initializer(1.))
        b = tf.get_variable('b', [num_filters], dtype=tf.float32, initializer=tf.constant_initializer(0.))

        V_norm = tf.nn.l2_normalize(V, [0, 1, 2])
        x = tf.nn.conv2d(x, V_norm, [1] + stride + [1], pad)
        ### weighted normalization
        if init:
            mean, var = tf.nn.moments(x, [0, 1, 2])
            g = tf.assign(g, init_scale / tf.sqrt(var + 1e-10))
            b = tf.assign(b, -mean * g)
        x = tf.reshape(g, [1, 1, 1, num_filters]) * x + tf.reshape(b, [1, 1, 1, num_filters])
        if scale == True:
            scale = tf.Variable(initial_value=0.0, trainable=True, name="rezero_scale")
            x = scale*x
        return x


@add_arg_scope
def deconv2d(x, num_filters, filter_size=[3, 3], stride=[1, 1], pad='SAME', init_scale=1., counters={}, init=False,
             **kwargs):
    ''' weight normalization transposed convolutional layer '''
    num_filters = int(num_filters)
    name = get_name('deconv2d', counters)
    xs = int_shape(x)
    strides = [1] + stride + [1]
    if pad == 'SAME':
        target_shape = [xs[0], xs[1] * stride[0],
                        xs[2] * stride[1], num_filters]
    else:
        target_shape = [xs[0], xs[1] * stride[0] + filter_size[0] -
                        1, xs[2] * stride[1] + filter_size[1] - 1, num_filters]
    with tf.variable_scope(name):
        V = tf.get_variable('V',
                            filter_size + [num_filters, xs[-1]],
                            tf.float32,
                            tf.random_normal_initializer(0, 0.05))
        g = tf.get_variable('g', [num_filters], dtype=tf.float32, initializer=tf.constant_initializer(1.))
        b = tf.get_variable('b', [num_filters], dtype=tf.float32, initializer=tf.constant_initializer(0.))

        V_norm = tf.nn.l2_normalize(V, [0, 1, 3])
        x = tf.nn.conv2d_transpose(x, V_norm, target_shape, [1] + stride + [1], pad)
        ### weighted normalization
        if init:
            mean, var = tf.nn.moments(x, [0, 1, 2])
            g = tf.assign(g, init_scale / tf.sqrt(var + 1e-10))
            b = tf.assign(b, -mean * g)
        x = tf.reshape(g, [1, 1, 1, num_filters]) * x + tf.reshape(b, [1, 1, 1, num_filters])

        return x


@add_arg_scope
def activate(x, activation, **kwargs):
    if activation == None:
        return x
    elif activation == "relu":
        return tf.nn.relu(x)
    else:
        raise NotImplemented(activation)


def nin(x, num_units):
    x = conv2d(x, num_units)
    return x 


def downsample(x, num_units):
    x = conv2d(x, num_units)
    x = activate(x)
    x = tf.layers.max_pooling2d(
        x,
        pool_size=2,
        strides=2,
        padding='SAME',
        data_format='channels_last',
        name='pool',
    )

    return x


def upsample(x, num_units, x_up=None, method="relinear"):

    ###Requiring padding
    if method == "conv_transposed":
        return deconv2d(x, num_units, stride=[2, 2])
    elif method == "subpixel":
        x = conv2d(x, 4 * num_units)
        x = tf.depth_to_space(x, 2)
        return x

    elif method == 'relinear':
        x = conv2d(x, num_units)
        x = tf.image.resize_bilinear(
            x,
            x_up.shape[1:3],
            align_corners=True,
        )
        return x


@add_arg_scope
def residual_block(flag, x, a=None, conv=conv2d, **kwargs):
    """Slight variation of original."""
    xs = int_shape(x)
    num_filters = xs[-1]

    residual = x
    if a is not None:
        a = nin(activate(a), num_filters)
        residual = tf.concat([residual, a], axis=-1)

    #Pre-activation resnet block
    residual = activate(residual)
    residual = tf.nn.dropout(residual, keep_prob= 1.0 - flag.dropout)
    residual = conv(residual, num_filters)
    residual = tf.layers.batch_normalization(residual, momentum=0.99, epsilon=0.001, training=flag.use_batch_statistics)

    return x + residual


def post_enc(
        x, c, flag, init=False, n_residual_blocks=2, activation="relu"):

    with model_arg_scope(
            init=init, dropout_p=flag.dropout, activation=activation):
        # outputs
        hs = []
        # prepare input
        xc = tf.concat([x, c], axis=-1)
        h = nin(xc, flag.init_filters)
        n_filters = flag.init_filters
        for l in range(flag.n_scales):
            # level module
            for i in range(n_residual_blocks):
                h = residual_block(flag, h)
                hs.append(h)
            # prepare input to next level
            if l + 1 < flag.n_scales:
                n_filters = min(2 * n_filters, flag.max_filters)
                h = downsample(h, n_filters)
        return hs



def prior_enc(
        c, flag, init=False, n_residual_blocks=2, activation="relu"):
    
    with model_arg_scope(
            init=init, dropout_p=flag.dropout, activation=activation):
        # outputs
        hs = []
        # prepare input
        h = nin(c, flag.init_filters)
        n_filters = flag.init_filters
        for l in range(flag.n_scales):
            # level module
            for i in range(n_residual_blocks):
                h = residual_block(flag, h)
                hs.append(h)
            # prepare input to next level
            if l + 1 < flag.n_scales:
                n_filters = min(2 * n_filters, flag.max_filters)
                h = downsample(h, n_filters)
        return hs


def dec(enc_feature, latent_z, latent_mean, sample, flag, init=False, n_residual_blocks=2, activation="relu"):
    assert n_residual_blocks % 2 == 0

    enc_feature = list(enc_feature)
    with model_arg_scope(
            init=init, dropout_p=flag.dropout, activation=activation):
        flag.init_filters = enc_feature[-1].shape.as_list()[-1]
        dec_temp = nin(enc_feature[-1], flag.init_filters)


        for l in range(flag.n_scales):
            for i in range(n_residual_blocks // 2):
                dec_temp = residual_block(flag, dec_temp, enc_feature.pop())
                #res_list.append(dec_temp)

            if flag.latent_scales_dims[l] != 0:  # if l < 2
                ###Sample or just use deterministic mean
                if sample[l]==False:
                    z_prior = latent_mean.pop(0)
                elif sample[l]==True:
                    z_prior = latent_z.pop(0)

                
                ### Concat latent vectors
                n_h_channels = dec_temp.shape.as_list()[-1]
                dec_temp = tf.concat([dec_temp, z_prior], axis=-1)
                dec_temp = nin(dec_temp, n_h_channels)

                enc_temp = enc_feature.pop()
                dec_temp = residual_block(flag, dec_temp, enc_temp)

                if l + 1 < flag.n_scales:
                    flag.init_filters = enc_feature[-1].shape.as_list()[-1]
                    dec_temp = upsample(dec_temp, flag.init_filters, x_up=enc_feature[-1])
            else:
                dec_temp = residual_block(flag, dec_temp, enc_feature.pop())
                if l + 1 < flag.n_scales:
                    flag.init_filters = enc_feature[-1].shape.as_list()[-1]
                    dec_temp = upsample(dec_temp, flag.init_filters, x_up=enc_feature[-1])


    return dec_temp


def latent_encoder(features, flag):
    latent_mv = []
    latent_z = []
    for l in range(len(flag.latent_scales_dims)):
        if flag.latent_scales_dims[l] != 0:  # if l < 2
            mean_vae = latent_parameters(features[-(2*l+1)], flag.latent_scales_dims[l])
            latent_mv.append(mean_vae)
            z = latent_sample(mean_vae)
            latent_z.append(z)
    return latent_z, latent_mv




def latent_parameters(
        h, num_filters=32, init=False, **kwargs):
    return conv2d(h, num_filters)


def latent_kl(q, p):
    mean1 = q
    mean2 = p

    kl = 0.5 * tf.square(mean2 - mean1)
    kl = tf.reduce_sum(kl, axis=[1, 2, 3])
    kl = tf.reduce_mean(kl)
    return kl


def latent_sample(p):
    mean = p
    stddev = 1.0
    eps = tf.random_normal(mean.shape, mean=0.0, stddev=1.0)
    return mean + stddev * eps
