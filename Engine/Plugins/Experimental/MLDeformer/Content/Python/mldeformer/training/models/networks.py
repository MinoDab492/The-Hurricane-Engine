# -*- coding: utf-8 -*-
"""
This module creates and initializes networks, defines learn. rate schedulers,
and set normalization and activation functions.

Copyright Epic Games, Inc. All Rights Reserved.
"""

import functools
import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler

from . import generator_networks, custom_layers


###############################################################################
# Helper Functions
###############################################################################

def get_norm_layer_1d(norm_type='instance'):
    """Return a 1d normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm1d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm1d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = lambda x: custom_layers.Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)

    return norm_layer


def get_nonlinear_activation(activation_type='ReLu', inplace=True):
    """Return a non-linear activation function

    Parameters:
        activation_type (str) -- the name of the non-linear activation: relu | lrelu | sigmoid | tanh | none
    """
    if activation_type == 'relu':
        nonlinearity = functools.partial(nn.ReLU, inplace=inplace)
    elif activation_type == 'lrelu':
        nonlinearity = functools.partial(nn.LeakyReLU, inplace=inplace)
    elif activation_type == 'sigmoid':
        nonlinearity = functools.partial(nn.Sigmoid)
    elif activation_type == 'tanh':
        nonlinearity = functools.partial(nn.Tanh)
    elif activation_type == 'none':
        nonlinearity = lambda x: custom_layers.Identity()
    else:
        raise NotImplementedError('Non-linear activation [%s] is not found' % activation_type)

    return nonlinearity


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | multiplicative | step | plateau | cosine | none

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def linear_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=linear_rule)
    elif opt.lr_policy == 'multiplicative':
        def multiplicative_rule(epoch):
            if epoch >= opt.niter:
                return opt.lr_gamma
            return 1.0

        scheduler = lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=multiplicative_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    elif opt.lr_policy == 'none':
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=1.0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)

    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    xavier is a good initialization method; normal and kaiming might work better 
    in some cases. Feel free to experiment yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
            'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('Initializing networks parameters with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        if len(gpu_ids) > 1:
            net = nn.DataParallel(net, gpu_ids)  # multi-GPUs

    init_weights(net, init_type, init_gain=init_gain)
    return net


###############################################################################
# Network definitions
###############################################################################

def define_DeepDeform(in_nc, out_nc, batch_size, net_type, num_linear_blocks=1,
                      linear_filter_factor=1, norm='none', activation='tanh', use_dropout=False,
                      init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a deep deformation generator network with fully connected linear layers.

    Parameters:
        in_nc (int)                -- number of elements of the input tensor
        out_nc (int)               -- number of elements of the output tensor
        batch_size (int)           -- batch size.
        net_type (str)             -- architecture's name: deep_deform_L-C, where L,C is the number of input linear and convolution filters
        num_linear_blocks (int)    -- number of hidden linear layers (blocks)
        linear_filter_factor (int) -- factor applied to the number of input linear filters to obtain the number of output linear filters
        norm (str)                 -- normalization layer used in the network: batch | instance | none
        activation (str)           -- non-linear activation function used in the network: relu | lrelu | sigmoid | tanh | none
        use_dropout (bool)         -- if dropout layers will be used
        init_type (str)            -- initialization method
        init_gain (float)          -- scaling factor for normal, xavier and orthogonal
        gpu_ids (int list)         -- which GPUs the network runs on: e.g., 0,1,2

    Returns:
        Generator network

    The implementation is an extension to the paper Fast and Deep Deformation Approximations.


    Note: The generator weights are initialized by <init_net>.
    """
    net = None
    norm_layer = get_norm_layer_1d(norm_type=norm)
    nonlinearity = get_nonlinear_activation(activation_type=activation)

    num_linear_units = int(net_type.split('_')[-1])
    if 'deep_deform' in net_type:
        print('Network dimensionality: Input params.: {} -- Output deltas: {}'.format(in_nc, out_nc))
        print('Batch size: {}'.format(batch_size))
        print('Number of hidden units (1st layer): {}'.format(num_linear_units))
        print('Number of linear layers: {}'.format(num_linear_blocks))
        print('Linear filter factor: {}'.format(linear_filter_factor))
        print('Norm. layer: {}'.format(norm))
        print('Activation function: {}'.format(activation))
        print('Use dropout: {}'.format(use_dropout))
        net = generator_networks.NlayerDeepDeformNet(in_nc, out_nc, batch_size, num_linear_blocks,
                                                     num_linear_filters=num_linear_units,
                                                     linear_factor=linear_filter_factor,
                                                     norm_layer=norm_layer, nonlinearity=nonlinearity,
                                                     use_dropout=use_dropout)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % net_type)

    return init_net(net, init_type, init_gain, gpu_ids)
