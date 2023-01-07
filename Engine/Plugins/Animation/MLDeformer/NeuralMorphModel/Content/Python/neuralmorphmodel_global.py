# -*- coding: utf-8 -*-
'''
Copyright Epic Games, Inc. All Rights Reserved.
'''
from importlib import reload
import torch
import math
import nmm_shared


class MLP(torch.nn.Module):
    def __init__(self,
                 input_layers: list,
                 activation_function=torch.nn.ELU):

        # Simple single mlp that takes an input of size <input_layers[0]> and
        # outputs a result of size <input_layers[-1]>
        super().__init__()

        self.layers = torch.nn.Sequential()
        for k in range(0, len(input_layers) - 1):
            self.layers.add_module('linear_%i' % k, torch.nn.Linear(input_layers[k], input_layers[k + 1]))
            if activation_function is not None:
                self.layers.add_module('activation_%i' % k, activation_function())

    def forward(self, x):
        return self.layers.forward(x)


class Corrective(torch.nn.Module):
    def __init__(self,
                 num_vertices,
                 in_features,
                 out_shapes,
                 hidden_layer_shape,
                 activation_function=torch.nn.ELU,
                 input_mean=None,
                 input_std=None,
                 output_mean=None,
                 output_std=None):

        # Takes as input a vector of size <in_features> and runs this through a ssimple MLP to generate
        # <out_shapes> morph target coefficients.
        super().__init__()
        self.num_vertices = num_vertices
        self.in_features = in_features
        self.out_shapes = out_shapes
        self.input_mean = input_mean
        self.input_std = input_std
        self.output_mean = output_mean
        self.output_std = output_std
        self.num_morph_targets = out_shapes

        # Create a set of empty morph targets.
        self.morph_target_matrix = torch.nn.parameter.Parameter(
            torch.zeros((self.num_vertices * 3, self.num_morph_targets)))
        torch.nn.init.kaiming_uniform_(self.morph_target_matrix, a=math.sqrt(5))

        # Create a separate tiny MLP for each joint, with it's own morph targets.
        network_layers = [self.in_features] + hidden_layer_shape + [self.out_shapes]
        self.network = MLP(input_layers=network_layers,
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

        # Do the main network.
        morph_target_coefficients = self.network.forward(x)

        if self.training:
            return self.get_deltas(morph_target_coefficients)
        else:
            return morph_target_coefficients


# Create the neural network structure.
def create_network(num_vertices,
                   num_features,
                   num_morph_targets,
                   num_bones,
                   num_hidden_layers,
                   num_units_per_hidden_layer,
                   shared_bone_indices,
                   device,
                   input_mean=None,
                   input_std=None,
                   output_mean=None,
                   output_std=None):
    hidden_layers = list()
    for i in range(num_hidden_layers):
        hidden_layers.append(num_units_per_hidden_layer)

    return Corrective(num_vertices=num_vertices,
                      in_features=num_features,
                      out_shapes=num_morph_targets,
                      hidden_layer_shape=hidden_layers,
                      input_mean=input_mean,
                      input_std=input_std,
                      output_mean=output_mean,
                      output_std=output_std).to(device)


# Main network training function, executed when we click the Train button in the UI.
# This is launched from inside the init_unreal.py file.
def train(training_model):
    reload(nmm_shared)
    print('Training global neural morph model')
    return nmm_shared.train(training_model, create_network, include_curves=True)
