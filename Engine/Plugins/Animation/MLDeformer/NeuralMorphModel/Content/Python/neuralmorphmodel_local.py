# -*- coding: utf-8 -*-
'''
Copyright Epic Games, Inc. All Rights Reserved.
'''
from importlib import reload
import torch
import math
import nmm_shared
import torch.nn as nn


class BlockLinear(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 blocks: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.blocks = blocks
        self.weight = nn.Parameter(torch.empty((blocks, out_features, in_features)))
        self.bias = nn.Parameter(torch.empty(blocks, out_features, 1))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return torch.matmul(self.weight, x) + self.bias


class MultiMLP(torch.nn.Module):
    def __init__(self,
                 input_layers: list,
                 num_blocks: int,
                 activation_function=torch.nn.ELU):

        # Block based mlp that takes an input of size <num_blocks x input_layers[0]> and
        # outputs a result of size <num_blocks x input_layers[-1]>
        super().__init__()
        self.num_blocks = num_blocks
        self.block_input_features = input_layers[0]
        self.input_features = num_blocks * input_layers[0]
        self.output_features = num_blocks * input_layers[-1]
        self.layers = torch.nn.Sequential()
        for k in range(0, len(input_layers) - 1):
            self.layers.add_module('linear_%i' % k,
                                   BlockLinear(input_layers[k], input_layers[k + 1], num_blocks))
            if activation_function is not None:
                self.layers.add_module('activation_%i' % k, activation_function())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = list(x.shape)[:-1]
        return self.layers.forward(x.reshape(shape + [self.num_blocks, -1, 1])).reshape(shape + [-1])


class Corrective(torch.nn.Module):
    def __init__(self,
                 num_vertices,
                 in_features_per_block,
                 out_shapes_per_block,
                 num_blocks,
                 hidden_layer_shape=[4, ],
                 activation_function=torch.nn.ELU,
                 shared_features=None,
                 input_mean=None,
                 input_std=None,
                 output_mean=None,
                 output_std=None):

        # Takes as input a vector of size <num_blocks x in_features_per_block> and runs this through
        # a small block base MLP to generate <num_blocks x out_shapes_per_block> morph target coefficients.
        # Additional will take a tensor of <m x n> shared blocks which will create an additional
        # <m x out_shapes_per_block> morph target coefficients concatenated.
        super().__init__()
        self.num_blocks = num_blocks
        self.num_shared_blocks = 0
        self.shared_features = shared_features
        if self.shared_features is not None:
            self.num_shared_blocks = shared_features.shape[0]
        self.num_vertices = num_vertices
        self.in_features_per_block = in_features_per_block
        self.out_shapes_per_block = out_shapes_per_block
        self.input_mean = input_mean
        self.input_std = input_std
        self.output_mean = output_mean
        self.output_std = output_std
        self.num_morph_targets = (self.num_blocks + self.num_shared_blocks) * out_shapes_per_block

        # Create a set of empty morph targets.
        self.morph_target_matrix = torch.nn.parameter.Parameter(
            torch.zeros((self.num_vertices * 3, self.num_morph_targets)))
        torch.nn.init.kaiming_uniform_(self.morph_target_matrix, a=math.sqrt(5))

        # Create a separate tiny MLP for each joint, with it's own morph targets.
        network_layers = [self.in_features_per_block] + hidden_layer_shape + [self.out_shapes_per_block]
        self.network = MultiMLP(input_layers=network_layers,
                                num_blocks=self.num_blocks,
                                activation_function=activation_function)

        # Also create an additional multiMLP for the shared blocks where the 
        # morph targets are controlled by multiple inputs.
        if self.shared_features is not None:
            shared_input_size = self.shared_features.shape[1]
            shared_layers = [shared_input_size * self.in_features_per_block] + hidden_layer_shape + [
                self.out_shapes_per_block]
            self.shared_network = MultiMLP(input_layers=shared_layers,
                                           num_blocks=self.num_shared_blocks,
                                           activation_function=activation_function)

    def get_deltas(self, morph_target_coefficients):
        assert morph_target_coefficients.shape[-1] == self.morph_target_matrix.shape[-1]

        # Calculate rest meshes according to the morph target coefficients.
        deltas = torch.matmul(self.morph_target_matrix, morph_target_coefficients.unsqueeze(-1)).squeeze(-1)

        if self.output_mean is not None and self.output_std is not None:
            return deltas * self.output_std + self.output_mean
        else:
            return deltas

    def forward(self, x_in):
        # Normalize the input if we have the information.
        if self.input_mean is not None and self.input_std is not None:
            x = (x_in - self.input_mean) / self.input_std
        else:
            x = x_in

        # First do the main network.
        morph_target_coefficients = self.network.forward(x)

        # Now the coefficients for all shared joint networks.
        if self.shared_features is not None:
            shared_target_coefficients = \
                self.shared_network.forward(
                    x.reshape(list(x.shape)[:-1] + [-1, self.in_features_per_block])
                    [..., self.shared_features, :].reshape(x.shape[0], -1))
            morph_target_coefficients = torch.cat([morph_target_coefficients, shared_target_coefficients], dim=-1)

        if self.training:
            return self.get_deltas(morph_target_coefficients)
        else:
            return morph_target_coefficients


# Create the neural network structure.
def create_network(num_vertices,
                   num_inputs,
                   num_morph_targets_per_bone,
                   num_bones,
                   num_hidden_layers,
                   num_units_per_hidden_layer,
                   shared_bone_indices,
                   device,
                   input_mean=None,
                   input_std=None,
                   output_mean=None,
                   output_std=None):
    shared_bones = torch.LongTensor(shared_bone_indices) if shared_bone_indices else None

    hidden_layers = list()
    for i in range(num_hidden_layers):
        hidden_layers.append(num_units_per_hidden_layer)

    return Corrective(num_vertices=num_vertices,
                      in_features_per_block=6,
                      out_shapes_per_block=num_morph_targets_per_bone,
                      num_blocks=num_bones,
                      shared_features=shared_bones,
                      hidden_layer_shape=hidden_layers,
                      input_mean=input_mean,
                      input_std=input_std,
                      output_mean=output_mean,
                      output_std=output_std).to(device)


# Main network training function, executed when we click the Train button in the UI.
# This is launched from inside the init_unreal.py file.
def train(training_model):
    reload(nmm_shared)
    print('Training local neural morph model')
    return nmm_shared.train(training_model, create_network, include_curves=False)
