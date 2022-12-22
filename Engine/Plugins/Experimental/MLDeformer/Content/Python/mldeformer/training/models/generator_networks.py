# -*- coding: utf-8 -*-
"""
This module creates generator networks (and its building blocks).

Copyright Epic Games, Inc. All Rights Reserved.
"""

import functools
import torch.nn as nn

from . import custom_layers


def activate_linear_bias(norm_layer, batch_size, num_channels):
    """Check is the bias must be activated in the linear function.
    Parameters:
        norm_layer   -- 1d normalization layer
        batch_size   -- batch size
        num_channels -- number of channels
    Return:
        Boolean variable
        """
    use_bias = True
    if norm_layer == nn.BatchNorm1d:
        if batch_size > 1 or num_channels > 1:
            use_bias = False
    elif norm_layer == nn.InstanceNorm1d:
        if num_channels > 1:
            use_bias = False
    return use_bias


class NlayerDeepDeformNet(nn.Module):
    """Create a N-layer deep deformation network, consisting of fully connected layers, to convert parameters into flattened 3D deformations"""

    def __init__(self, input_nc, output_nc, batch_size, num_linear_blocks,
                 num_linear_filters=128, linear_factor=1, norm_layer=nn.BatchNorm1d,
                 nonlinearity=nn.Tanh, use_dropout=False):
        """Construct a N-layer deep deformation network
        Parameters:
            input_nc (int)           -- number of elements of the input tensor.
            output_nc (int)          -- number of elements of the output tensor.
            batch_size (int)         -- batch size.
            num_linear_blocks (int)  -- the number of hidden linear layers to generate the output points.
            num_linear_filters (int) -- the number of hidden linear units in the 1st linear layer.
            linear_factor (int)      -- multiplicative factor applied to the output filter count.
            norm_layer               -- 1d normalization layer
            nonlinearity             -- nonlinear activation function
            use_dropout              -- whether dropout must be used in the hidden layers

        Construct N-layer generator network from left (input) to right (output).
        """
        # Construct N-layer FC net
        super(NlayerDeepDeformNet, self).__init__()

        # Create input linear block.
        num_in_filters = input_nc
        num_out_filters = num_linear_filters
        fc_net = DDBlock.add_fc_module(num_out_filters, num_in_filters, batch_size, outermost=False,
                                       norm_layer=norm_layer, nonlinearity=nonlinearity, use_dropout=use_dropout)

        # Add intermediate linear blocks.
        for i in range(1, num_linear_blocks + 1):
            num_in_filters = num_out_filters
            num_out_filters = max(int(num_out_filters * linear_factor), 8)
            if i == 1:
                fc_net += DDBlock.add_fc_module(num_out_filters, num_in_filters, batch_size, outermost=False,
                                                norm_layer=norm_layer, nonlinearity=nonlinearity,
                                                use_dropout=use_dropout)
            else:
                fc_net += DDBlock.add_fc_module(num_out_filters, num_in_filters, batch_size, outermost=False,
                                                norm_layer=norm_layer, nonlinearity=nonlinearity,
                                                use_dropout=use_dropout)

        # Create output linear block.
        num_in_filters = num_out_filters
        num_out_filters = output_nc
        fc_net += DDBlock.add_fc_module(num_out_filters, num_in_filters, batch_size, outermost=True,
                                        norm_layer=norm_layer, nonlinearity=nonlinearity, use_dropout=False)

        self.fc_seq_net = nn.Sequential(*fc_net)

    def forward(self, in_tensor):
        """Forward function
        Parameters:
            in_tensor (tensor) -- input tensor.
        """
        out_tensor = self.fc_seq_net(in_tensor)
        return out_tensor


class DDBlock(nn.Module):
    """Stacks a fully connected block in the deep deformation network.
        |-- |net| -- FC DD block --|
    """

    @staticmethod
    def add_fc_module(outer_nc, inner_nc, batch_size, outermost=False,
                      norm_layer=nn.BatchNorm1d, nonlinearity=nn.Tanh, use_dropout=False):
        """Add a fully connected block to the network.

        Parameters:
            outer_nc (int)      -- number of filters in the outer layer
            inner_nc (int)      -- number of filters in the inner layer
            batch_size (int)    -- batch size
            outermost (bool)    -- whether is the outermost layer
            norm_layer          -- normalization layer
            nonlinearity        -- non linearity activation
            user_dropout (bool) -- if use dropout layers.
        """
        if outermost:
            use_bias = True
        else:
            # Check if the linear layer must use bias.
            if type(norm_layer) == functools.partial:
                use_bias = activate_linear_bias(norm_layer.func, batch_size, 1)
            else:
                use_bias = activate_linear_bias(norm_layer, batch_size, 1)

        # Add linear layer, either single or multi-channel.
        fc = nn.Linear(inner_nc, outer_nc, bias=use_bias)

        # Create and return FC block.
        if outermost:
            # Last block is just a plain FC layer.
            return [fc]
        else:
            # Add norm layer
            if use_bias:
                norm_layer = lambda x: custom_layers.Identity()
                norm = norm_layer(outer_nc)
            else:
                norm = norm_layer(outer_nc, affine=True)

            # Add non-linearity
            nla = nonlinearity()

            # Create and return block.
            if use_dropout:
                # Add dropout layer to block.
                return [fc, norm, nla, nn.Dropout(0.5)]
            else:
                return [fc, norm, nla]
