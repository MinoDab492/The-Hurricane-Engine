# -*- coding: utf-8 -*-
"""
This module contains different flavors for visualizing logs.

Copyright Epic Games, Inc. All Rights Reserved.
"""

import numpy as np
import os
import sys
import time
from subprocess import Popen, PIPE
from torch.autograd import Variable

from mldeformer.utils.io import os_utils

if sys.version_info[0] == 2:
    VisdomExceptionBase = Exception
else:
    VisdomExceptionBase = ConnectionError


class Visualizer():
    """Generic model visualizer that can print/save results logs during training."""

    def __init__(self, opt):
        """Initialize the Visualizer class

        Parameters:
            opt -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        # Cache the option
        self.opt = opt

        # Create log file to store training losses.
        if not opt.no_log:
            self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
            with open(self.log_name, 'a') as log_file:
                now = time.strftime("%c")
                log_file.write('================ Training Loss ({}) ================\n'.format(now))

    def reset(self):
        pass

    # Print losses. The output has the same format as |losses| of plot_current_losses.
    def print_current_losses(self, epoch, iters, losses, t_comp, t_data=None):
        """Print current losses on console; also save the losses to the disk if the user wants the log.

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        if t_data is not None:
            message = '(epoch: {:d}, iters: {:d}, batch eval time: {:.4f}s, batch load time: {:.4f}s) '.format(epoch,
                                                                                                               iters,
                                                                                                               t_comp,
                                                                                                               t_data)
        else:
            message = '(epoch: {:d}, iters: {:d}, batch eval time: {:.4f}s) '.format(epoch, iters, t_comp)
        for k, v in losses.items():
            message += '{}: {:.4f} '.format(k, v)

        # Print message.
        print(message)
        if not self.opt.no_log:
            # Save message to file.
            with open(self.log_name, 'a') as log_file:
                log_file.write('{}\n'.format(message))
