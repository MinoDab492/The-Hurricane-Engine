# -*- coding: utf-8 -*-
"""
TrainOptions class.

Copyright Epic Games, Inc. All Rights Reserved.
"""

from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def __init__(self):
        """Define default options for training."""
        BaseOptions.__init__(self)

        ##############################
        ## Visualization parameters ##
        ##############################
        # Whether stats will be logged/saved to file. Default is no_log.
        # Activate if you want to keep track of your training attempts.
        self.no_log = True

        ################################################
        ## Parameters for saving and loading networks ##
        ################################################
        # Epoch that will be loaded. This is handy if we want to resume the training at some point.
        self.epoch = 'latest'

        # Frequency of saving checkpoints at the end of epochs.
        self.save_epoch_freq = 5
        
        # Save epochs networks?
        self.save_epochs = False

        # Load the latest model if we continue training. Default is False (train from scratch).
        self.continue_train = False

        # Starting epoch count. Handy if we continue training from an existing checkpoint. Default is 1 (train from scratch).
        self.epoch_count = 1

        #########################
        ## Caching parameters ##
        #########################
        self.memory_cache_in_megabytes = 500
        
        #########################
        ## Profiling parameters ##
        #########################
        self.profile = False

        #########################
        ## Training parameters ##
        #########################
        # Loss type: L1 or shrinkage. Default is L1.
        self.loss_type = 'L1'

        # Network initialization type: normal, xavier, kaiming, orthogonal. Default is xavier, which works well in practice.
        # Other initialization might work well for certain assets and custom architectures.
        self.init_type = 'xavier'

        # Scaling factor applied to normal, xavier or orthogonal initialization. Default 0.02 (found empirically).
        self.init_gain = 0.02

        # Number of iterations (epochs) that will be run with initial learning rate.
        self.niter = 10

        # Number of iterations (epochs) that will decay the learning rate.
        self.niter_decay = 20

        # Momentum term of Adam optimizer. Keep this value relatively high; otherwise,
        # gradients are accumulated over large batches harming convergence and accuracy.
        self.beta1 = 0.9

        # Learning rate used by optimizer. Value between 1e-3 and 1-e4 work well in practice. Default is 0.0005.
        self.lr = 0.0005

        # Learning rate policy: linear, multiplicative, step, plateau, cosine, none. Default is linear.
        self.lr_policy = 'linear'

        # Learning rate factor. Only applicable is policy is multiplicative. Default is 0.95.
        self.lr_gamma = 0.95

        # Learning rate decay iterations. Only applicable if polciy is step. Default is 5.
        self.lr_decay_iters = 5

        ###############################
        ## Shrinkage loss parameters ##
        ###############################
        # Shrinkage localization (or threshold). Error tolerance to start applying the shrinkage loss.
        # Default is 0.01 for normalize deltas.
        self.shrink_loc = 0.01

        # Shrinkage loss speed. How fast we want to penalize error beyond the localization (or threshold).
        self.shrink_speed = 10.0

        self.isTrain = True

    def print_options(self):
        """Print all member variables, both base and training options."""
        message = '\n############################################\n'
        message += '############# Training options ############\n'
        message += '###########################################\n'
        for k, v in sorted(vars(self).items()):
            comment = ''
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)
