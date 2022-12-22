# -*- coding: utf-8 -*-
"""
DeepDeform class.

Copyright Epic Games, Inc. All Rights Reserved.
"""

import torch

from mldeformer.training.loss.loss_functions import L2pLoss
from mldeformer.training.loss.loss_functions import ShrinkageLoss
from . import custom_layers
from . import generator_networks
from . import networks
from .base_model import BaseModel


class DeepDeformModel(BaseModel):
    """ This class implements the deep deformation model that estimates dense 3D delta corrections for the inaccurate mesh skinning.

    The model training requires '--dataset_mode points' dataset.
    It can use a '--netG deep_deform_YYY' deep deform network, where YYY={128,256,512,1024}
 
    The network architecture and loss functions are similar to that of fast and deep deformation approximations
    http://people.eecs.berkeley.edu/~stephen.w.bailey/publications/fastapprox/DeformationApproximation.pdf
    """

    def __init__(self, opt):
        """Initialize this deep_deform class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - (required) call the initialization function of BaseModel
        - define loss function, output meshes, model names, and optimizers
        """
        BaseModel.__init__(self, opt)
        # Specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = [opt.loss_type]

        # Specify the meshes want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.mesh_names = ['pred_vec', 'out_vec']

        # Specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names = ['G']
        self.converted_model_names = ['G']

        # Loss scaling factor
        self.data_weight = 100.0

        # Define network
        self.netG = networks.define_DeepDeform(opt.input_nc, opt.output_nc, opt.batch_size, opt.netG,
                                               opt.n_linear_layers_G, opt.linear_factor, opt.norm, opt.activation,
                                               not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # Define loss functions.
            if opt.loss_type == 'L1':
                self.criterionLoss = L2pLoss(p=1.0)
            elif opt.loss_type == 'MSE':
                self.criterionLoss = torch.nn.MSELoss()
            elif opt.loss_type == 'Shrinkage':
                self.criterionLoss = ShrinkageLoss(shrink_speed=opt.shrink_speed, shrink_loc=opt.shrink_loc)

            # Initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer)

        # Our program will automatically call <model.setup> to define schedulers, load networks, and print networks

    def set_input(self, input_data):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input_data (dic): a dictionary that contains the data itself and its metadata information.
        """
        self.in_params = input_data['in_params'].to(self.device)
        self.out_vec = input_data['out_vec'].to(self.device)

        # Get data path.
        self.data_path = input_data['path']

    def forward(self):
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        # Generate output 3D deformation, computed as the sum between the input vec and delta vec.
        self.pred_vec = self.netG(self.in_params)

    def backward(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # Calculate the loss between the predictions and labels.
        if self.opt.loss_type == 'L1':
            self.loss_L1 = self.data_weight * self.criterionLoss(self.pred_vec, self.out_vec)
        elif self.opt.loss_type == 'MSE':
            self.loss_MSE = self.data_weight * self.criterionLoss(self.pred_vec, self.out_vec)
        elif self.opt.loss_type == 'Shrinkage':
            self.loss_Shrinkage = self.data_weight * self.criterionLoss(self.pred_vec, self.out_vec)

        # Compute the network's gradients.
        if self.opt.loss_type == 'L1':
            self.loss_L1.backward()
        elif self.opt.loss_type == 'MSE':
            self.loss_MSE.backward()
        elif self.opt.loss_type == 'Shrinkage':
            self.loss_Shrinkage.backward()

    def optimize_parameters(self):
        """Update network weights; it will be called in every training iteration."""
        # First call forward to calculate intermediate results
        self.forward()

        # Clear network G's existing gradients and compute gradients.
        self.optimizer.zero_grad()
        self.backward()

        # Update network's gradients.
        self.optimizer.step()

    def convert(self):
        """Convert the network into a module with a specific format."""
        if self.netG != None:
            # Remove unnecessary layers.
            net = torch.nn.Sequential()
            lin_id = 1
            act_id = 1
            norm_id = 1
            drop_id = 1
            if len(self.gpu_ids) > 1 and torch.cuda.is_available():
                children = self.netG.module.children()
            else:
                children = self.netG.children()

            for child in children:
                for layer in child.modules():
                    if not isinstance(layer, generator_networks.DDBlock) and \
                        not isinstance(layer, torch.nn.modules.Sequential) and \
                        not isinstance(layer, custom_layers.Identity):
                        if isinstance(layer, torch.nn.modules.Linear):
                            net.add_module('fc_{}'.format(lin_id), layer)
                            lin_id += 1
                        elif isinstance(layer, torch.nn.modules.BatchNorm1d):
                            net.add_module('batch_norm_{}'.format(norm_id), layer)
                            norm_id += 1
                        elif isinstance(layer, torch.nn.modules.InstanceNorm1d):
                            net.add_module('inst_norm_{}'.format(norm_id), layer)
                            norm_id += 1
                        elif isinstance(layer, torch.nn.modules.ReLU):
                            net.add_module('relu_{}'.format(act_id), layer)
                            act_id += 1
                        elif isinstance(layer, torch.nn.modules.LeakyReLU):
                            net.add_module('lrelu_{}'.format(act_id), layer)
                            act_id += 1
                        elif isinstance(layer, torch.nn.modules.Tanh):
                            net.add_module('tanh_{}'.format(act_id), layer)
                            act_id += 1
                        elif isinstance(layer, torch.nn.modules.Sigmoid):
                            net.add_module('sigmoid_{}'.format(act_id), layer)
                            act_id += 1
                        elif isinstance(layer, torch.nn.modules.Dropout):
                            net.add_module('drop_{}'.format(drop_id), layer)
                            drop_id += 1
                        else:
                            net.add_module(layer)
                            
            self.converted_netG = net

            # Assigns the model to self.netG so that it can be loaded
            # Please see <BaseModel.load_networks>
            setattr(self, 'converted_netG', self.converted_netG)  # store converted_netG in self.
