# The following code snippet is inspired and borrowed from the GazeML project (https://github.com/swook/GazeML).
# Author: Seonwook Park
from typing import Dict

import numpy as np
import scipy
import tensorflow as tf

from core import BaseDataSource, BaseModel
import util.gaze
import util.nn

class VGE(BaseModel):
    """Code from Deep Pictorial Gaze architecture as introduced in [Park et al. ECCV'18]."""

    def __init__(self, tensorflow_session=None, extra_tags=[], flag = None, **kwargs): # c1=1e-4, c2=1e-5
        """Specify VGE-specific parameters."""

        self._extra_tags = extra_tags
        self.flag = flag
        self.kl_weight = tf.placeholder(tf.float32, shape=(), name='kl_weights')
        super().__init__(tensorflow_session, **kwargs)

    _builder_num_feature_maps = 16
    _builder_num_residual_blocks = 1

    _dn_growth_rate = 8
    _dn_compression_factor = 0.5
    _dn_num_layers_per_block = (4, 4, 4, 4)
    _dn_num_dense_blocks = len(_dn_num_layers_per_block)

    @property
    def identifier(self):
        """Identifier for model based on data sources and parameters."""
        first_data_source = next(iter(self._train_data.values()))
        input_tensors = first_data_source.output_tensors
        if self._data_format == 'NHWC':
            _, eh, ew, _ = input_tensors['eye'].shape.as_list()
        else:
            _, _, eh, ew = input_tensors['eye'].shape.as_list()

        return 'VGE%s' % (
            '-'.join(self._extra_tags) if len(self._extra_tags) > 0 else '',
        )


    _column_of_ones = None
    _column_of_zeros = None

    def _augment_training_images(self, images, mode):
        if mode == 'test':
            return images
        with tf.variable_scope('augment'):
            if self._data_format == 'NCHW':
                images = tf.transpose(images, perm=[0, 2, 3, 1])
            n, h, w, _ = images.shape.as_list()
            if self._column_of_ones is None:
                self._column_of_ones = tf.ones((n, 1))
                self._column_of_zeros = tf.zeros((n, 1))
            transforms = tf.concat([
                self._column_of_ones,
                self._column_of_zeros,
                tf.truncated_normal((n, 1), mean=0, stddev=.05*w),
                self._column_of_zeros,
                self._column_of_ones,
                tf.truncated_normal((n, 1), mean=0, stddev=.05*h),
                self._column_of_zeros,
                self._column_of_zeros,
            ], axis=1)
            images = tf.contrib.image.transform(images, transforms, interpolation='BILINEAR')
            if self._data_format == 'NCHW':
                images = tf.transpose(images, perm=[0, 3, 1, 2])
        return images

    def build_model(self, data_sources: Dict[str, BaseDataSource], mode: str):
        
        """Build model."""
        data_source = next(iter(data_sources.values()))
        input_tensors = data_source.output_tensors
        x = input_tensors['eye']
        y1 = input_tensors['gazemaps'] if 'gazemaps' in input_tensors else None
        y2 = input_tensors['gaze'] if 'gaze' in input_tensors else None

        with tf.name_scope('trans'):
            if self._data_format == 'NCHW':
                yn = tf.transpose(y1, perm=[0, 2, 3, 1])
        outputs = {}
        loss_terms = {}
        metrics = {}

        # Lightly augment training data
        x = self._augment_training_images(x, mode)
        self.flag.use_batch_statistics = self.use_batch_statistics
        with (tf.variable_scope('vae')):
            with tf.variable_scope('pre'):
                xn = tf.transpose(x, perm=[0, 2, 3, 1])
            self.summary.feature_maps('x_in', x, data_format=self._data_format_longer)
            self.summary.feature_maps('y_in', y1, data_format=self._data_format_longer)

            with tf.variable_scope("prior_encoder"):
                gs = util.nn.prior_enc(xn, self.flag)
            with tf.variable_scope("prior_latent"):   
                zs_prior, m_v_prior = util.nn.latent_encoder(gs, self.flag)

            if mode == 'train':
                with tf.variable_scope("post_encoder"):
                    hs = util.nn.post_enc(xn, yn, self.flag)
                with tf.variable_scope("post_latent"): 
                    zs_posterior, m_v_post = util.nn.latent_encoder(hs, self.flag)


            with tf.variable_scope("GMAP"):
                with tf.variable_scope("decoder"):
                    last_dec = util.nn.dec(gs, zs_prior, m_v_prior, [True,True,True,True,True,True,True], self.flag)
                with tf.variable_scope("builder"):
                    gmap = self._build_after(tf.transpose(last_dec, perm=[0, 3, 1, 2]))


                # resize y1 GT
                yr = tf.transpose(yn, perm = [0, 3, 1, 2])

                # cross-entropy loss (more stable than l2 loss)
                metrics['gazemaps'] = -tf.reduce_mean(tf.reduce_sum(yr * tf.log(tf.clip_by_value(gmap, 1e-10, 1.0)),  # avoid NaN
                                            axis=[1, 2, 3]))
                

                if mode=='train':
                    #KL loss
                    kl_loss_list = []
                    kl_loss = tf.to_float(0.0)
                    for q, p in zip(m_v_post, m_v_prior):
                        loss_var = util.nn.latent_kl(q, p)
                        kl_loss_list.append(loss_var)
                        kl_loss += tf.reduce_mean(loss_var)
                    metrics['kl_loss'] = tf.reduce_sum(kl_loss)
                    outputs['kl_loss'] = kl_loss_list

                    #CVAE loss
                    metrics['gazemaps_ce'] =  metrics['gazemaps'] + self.flag.kl_w * metrics['kl_loss']

                x = gmap
                outputs['gazemaps'] = gmap

                self.summary.feature_maps('bottleneck', gmap, data_format=self._data_format_longer)

        with tf.variable_scope('densenet'):
            x = gmap
            for i in range(self._dn_num_dense_blocks):
                with tf.variable_scope('block%d' % (i + 1)):
                    x = self._apply_dense_block(x,
                                                num_layers=self._dn_num_layers_per_block[i])
                    if i == self._dn_num_dense_blocks - 1:
                        break
                with tf.variable_scope('trans%d' % (i + 1)):
                    x = self._apply_transition_layer(x)

            # Global average pooling
            with tf.variable_scope('post'):
                x = self._apply_bn(x)
                x = tf.nn.relu(x)
                if self._data_format == 'NCHW':
                    x = tf.reduce_mean(x, axis=[2, 3])
                else:
                    x = tf.reduce_mean(x, axis=[1, 2])
                x = tf.contrib.layers.flatten(x)

            # Output layer
            with tf.variable_scope('output'):
                gaze_direction = self._apply_fc(x, 2)


                if y2 is not None:
                    metrics['gaze_mse'] = tf.reduce_mean(tf.squared_difference(gaze_direction, y2))
                    metrics['gaze_ang'] = util.gaze.tensorflow_angular_error_from_pitchyaw(y2, gaze_direction)

        if yr is not None and y2 is not None and mode=='train':
            loss_terms['combined_loss'] = 1e-4*metrics['gazemaps_ce'] + metrics['gaze_mse']


        # Define outputs
        return outputs, loss_terms, metrics



    def _apply_conv(self, tensor, num_features, kernel_size=3, stride=1):
        return tf.layers.conv2d(
            tensor,
            num_features,
            kernel_size=kernel_size,
            strides=stride,
            padding='SAME',
            kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4),
            bias_initializer=tf.zeros_initializer(),
            data_format=self._data_format_longer,
            name='conv',
        )

    def _apply_fc(self, tensor, num_outputs):
        return tf.layers.dense(
            tensor,
            num_outputs,
            use_bias=True,
            kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4),
            bias_initializer=tf.zeros_initializer(),
            name='fc',
        )

    def _apply_pool(self, tensor, kernel_size=3, stride=2):
        tensor = tf.layers.max_pooling2d(
            tensor,
            pool_size=kernel_size,
            strides=stride,
            padding='SAME',
            data_format=self._data_format_longer,
            name='pool',
        )
        return tensor

    def _apply_bn(self, tensor):
        return tf.contrib.layers.batch_norm(
            tensor,
            scale=True,
            center=True,
            is_training=self.use_batch_statistics,
            trainable=True,
            data_format=self._data_format,
            updates_collections=None,
        )

    def _build_residual_block(self, x, num_in, num_out, name='res_block'):
        with tf.variable_scope(name):
            half_num_out = max(int(num_out/2), 1)
            c = x
            with tf.variable_scope('conv1'):
                c = tf.nn.relu(self._apply_bn(c))
                c = self._apply_conv(c, num_features=half_num_out, kernel_size=1, stride=1)
            with tf.variable_scope('conv2'):
                c = tf.nn.relu(self._apply_bn(c))
                c = self._apply_conv(c, num_features=half_num_out, kernel_size=3, stride=1)
            with tf.variable_scope('conv3'):
                c = tf.nn.relu(self._apply_bn(c))
                c = self._apply_conv(c, num_features=num_out, kernel_size=1, stride=1)
            with tf.variable_scope('skip'):
                if num_in == num_out:
                    s = tf.identity(x)
                else:
                    s = self._apply_conv(x, num_features=num_out, kernel_size=1, stride=1)
            x = c + s
        return x

    def _build_after(self, x_now):
        self._builder_num_feature_maps = x_now.shape.as_list()[1]
        with tf.variable_scope('after'):
            for j in range(self._builder_num_residual_blocks):
                x_now = self._build_residual_block(x_now, self._builder_num_feature_maps,
                                                   self._builder_num_feature_maps,
                                                   name='after_uv_%d' % (j + 1))
            x_now = self._apply_conv(x_now, self._builder_num_feature_maps, kernel_size=1, stride=1)
            x_now = self._apply_bn(x_now)
            x_now = tf.nn.relu(x_now)

            with tf.variable_scope('gmap'):
                gmap = self._apply_conv(x_now, 2, kernel_size=1, stride=1)

        # Perform softmax on gazemaps
        if self._data_format == 'NCHW':
            n, c, h, w = gmap.shape.as_list()
            gmap = tf.reshape(gmap, (n, -1))
            gmap = tf.nn.softmax(gmap)
            gmap = tf.reshape(gmap, (n, c, h, w))
        else:
            n, h, w, c = gmap.shape.as_list()
            gmap = tf.transpose(gmap, perm=[0, 3, 1, 2])
            gmap = tf.reshape(gmap, (n, -1))
            gmap = tf.nn.softmax(gmap)
            gmap = tf.reshape(gmap, (n, c, h, w))
            gmap = tf.transpose(gmap, perm=[0, 2, 3, 1])
        return gmap

    def _apply_dense_block(self, x, num_layers):
        assert isinstance(num_layers, int) and num_layers > 0
        c_index = 1 if self._data_format == 'NCHW' else 3
        x_prev = x
        for i in range(num_layers):
            with tf.variable_scope('layer%d' % (i + 1)):
                n = x.shape.as_list()[c_index]
                with tf.variable_scope('bottleneck'):
                    x = self._apply_composite_function(x,
                                                       num_features=min(n, 4*self._dn_growth_rate),
                                                       kernel_size=1)
                with tf.variable_scope('composite'):
                    x = self._apply_composite_function(x, num_features=self._dn_growth_rate,
                                                       kernel_size=3)
                if self._data_format == 'NCHW':
                    x = tf.concat([x, x_prev], axis=1)
                else:
                    x = tf.concat([x, x_prev], axis=-1)
                x_prev = x
        return x

    def _apply_transition_layer(self, x):
        c_index = 1 if self._data_format == 'NCHW' else 3
        x = self._apply_composite_function(
            x, num_features=int(self._dn_compression_factor * x.shape.as_list()[c_index]),
            kernel_size=1)
        x = tf.layers.average_pooling2d(x, pool_size=2, strides=2, padding='valid',
                                        data_format=self._data_format_longer)
        return x

    def _apply_composite_function(self, x, num_features=_dn_growth_rate, kernel_size=3):
        x = self._apply_bn(x)
        x = tf.nn.relu(x)
        x = self._apply_conv(x, num_features=num_features, kernel_size=kernel_size, stride=1)
        return x