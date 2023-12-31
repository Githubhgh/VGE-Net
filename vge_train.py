# The following code snippet is inspired and borrowed from the GazeML project (https://github.com/swook/GazeML).
# Author: Seonwook Park
"""Main script for training the VGE model for within-MPIIGaze evaluations."""
import argparse
import os
import coloredlogs
import tensorflow as tf
import numpy as np
import nni
import logging
from nni.utils import merge_parameter

import datetime
import pytz
import time



logger = logging.getLogger('VGE_AutoML')


def get_params():
    # Set global log level
    parser = argparse.ArgumentParser(description='Train the Deep Pictorial Gaze model.')
    parser.add_argument('-v', type=str, help='logging level', default='info',
                        choices=['debug', 'info', 'warning', 'error', 'critical'])
    args = parser.parse_args()
    coloredlogs.install(
        datefmt='%d/%m %H:%M',
        fmt='%(asctime)s %(levelname)s %(message)s',
        level=args.v.upper(),
    )
    ''' Get parameters from command line '''
    tf.flags.DEFINE_integer('batch_size', 64, 'batch size')
    tf.flags.DEFINE_integer('c1', 1, 'test_id')
    tf.flags.DEFINE_float('lr', 6e-4, 'learning rate')
    tf.flags.DEFINE_integer('init_filters', 64, 'init feature map') #If not use up-sample and down-sample, replace it with 256
    tf.flags.DEFINE_integer('max_filters', 256, 'max feature map')
    tf.flags.DEFINE_integer('n_scales', 7, 'scale for U-Net')
    tf.flags.DEFINE_multi_integer('latent_scales_dims', [1, 2, 0, 0, 0, 0, 0], 'u-net latent dim')
    tf.flags.DEFINE_float('kl_w', 10.0, 'kl loss weight')
    tf.flags.DEFINE_string('_data_format', 'NCHW', 'data format for eye image')
    tf.flags.DEFINE_bool('use_batch_statistics', True, 'using batch_norm or not')
    tf.flags.DEFINE_float('dropout', 0, 'dropout ratio')
    FLAGS = tf.flags.FLAGS

    return FLAGS


os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def train(FLAGS):
    from datetime import datetime
    import pytz

    from datetime import datetime, timedelta

    # 获取当前时间
    current_time = datetime.now()

    # Set global log level
    parser = argparse.ArgumentParser(description='Train the Deep Pictorial Gaze model.')
    parser.add_argument('-v', type=str, help='logging level', default='info',
                        choices=['debug', 'info', 'warning', 'error', 'critical'])
    args = parser.parse_args()

    coloredlogs.install(
        datefmt='%d/%m %H:%M',
        fmt='%(asctime)s %(levelname)s %(message)s',
        level=args.v.upper(),
    )

    for i in range(0, 15):
        if i != FLAGS.c1:
            print('pass id' + str(i))
            continue
        # Specify which people to train on, and which to test on
        person_id = 'p%02d' % i
        other_person_ids = ['p%02d' % j for j in range(15) if i != j]

        # Initialize Tensorflow session
        tf.reset_default_graph()
        tf.logging.set_verbosity(tf.logging.ERROR)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as session:

            # Define training data source
            from datasources import HDF5Source

            # Define model
            from models import VGE
            model = VGE(
                session,
                learning_schedule=[
                    {
                        'loss_terms_to_optimize': {
                            'combined_loss': ['vae', 'densenet'],
                        },
                        'metrics': ['gaze_mse', 'gaze_ang', 'kl_loss', 'gazemaps'],
                        'learning_rate': FLAGS.lr,
                    },

                ],
                extra_tags=[str(person_id) + '_' + str(current_time)],
                flag=FLAGS,
                # Data sources for training (and testing).
                train_data={
                    'mpi': HDF5Source(
                        session,
                        data_format='NCHW',
                        batch_size=FLAGS.batch_size,
                        keys_to_use=['train/' + s for s in other_person_ids],
                        hdf_path='datasets/MPIIGaze.h5',
                        #eye_image_shape=(96, 160), #original size, use that if you have enough computation resource.
                        eye_image_shape=(36, 60), #resize to save gpu memory
                        testing=False,
                        min_after_dequeue=1000,
                        staging=True,
                        shuffle=True,
                    ),
                },
                test_data={
                    'mpi': HDF5Source(
                        session,
                        data_format='NCHW',
                        batch_size=FLAGS.batch_size,
                        keys_to_use=['test/' + person_id],
                        hdf_path='datasets/MPIIGaze.h5',
                        #eye_image_shape=(96, 160), #original size, use that if you have enough computation resource.
                        eye_image_shape=(36, 60), #resize to save gpu memory
                        testing=True,
                    ),
                },
            )

            # Train this model for a set number of epochs
            model.train(
                num_epochs=20, now_leave_one_test=i #the epoches can be further longer to achieve stable performance
            )
            model.__del__()
            session.close()
            del session


if __name__ == '__main__':



    try:
        # get parameters form tuner
        tuner_params = nni.get_next_parameter()
        logger.debug(tuner_params)
        params = merge_parameter(get_params(), tuner_params)
        print(params)
        train(params)


    except Exception as exception:
        logger.exception(exception)
        raise
