# -*- coding: utf-8 -*-
"""
BaseOptions class.

Copyright Epic Games, Inc. All Rights Reserved.
"""


class BaseOptions():
    """This class defines options used during both training and test time."""

    def __init__(self):
        """Define default options for training and testing."""

        ####################
        ## I/O parameters ##
        ####################
        # Root path to store data.
        self.dataroot = ''
        
        # Folder where models are stored.
        self.checkpoints_dir = './checkpoints'
        
        # Name of the experiment. This is name decides the sub-folder where models are stored.
        self.name = 'MLDeformerModels'
        
        # Gpu ids: e.g. [0]  [0,1,2]. Use [] for CPU.
        self.gpu_ids = []

        ######################
        ## Model parameters ##
        ######################
        # Model that will be used to learn ML Deformer.
        self.model = 'deep_deform'
        
        # Input and output dimensionality. Default values.
        self.input_nc = 0
        self.output_nc = 0
        
        # Generator architeture. The suffix defines the number of hidden linear units.
        self.netG = self.model + '_256'
        
        # Number of hidden layers.
        self.n_linear_layers_G = 2
        
        # Linear factor. Incremental factor applied to number of hidden units. Default 1 (no increment).
        self.linear_factor = 1
        
        # Type of normalization function: none, instance or batch. Default is none.
        self.norm = 'none'
        
        # Activation function: relu, lrelu, tanh, sigmoid or none. Default is relu (works well in most cases).
        self.activation = 'relu'
        
        # If no dropout is applied to layer. Default is True (i.e., no dropout applied).
        self.no_dropout = True
        
        # Network precision: double, float, half. Default is float.
        # Half reduces memory comsumption but use it at your own risk (experimental).
        self.precision = 'float'

        ########################
        ## Dataset parameters ##
        ########################
        # Class to handle dataset. Default is unreal_deformer. The user can add custom dataset loader, e.g. parallel data loader.
        self.dataset_mode = 'unreal_deformer'
        
        # Whether smaple in the batch are taken sequentially from the sequence. Default is False.
        self.serial_batches = False
        
        # Number of threads. Default is 0 (do not change this value; windows can't parallelize while not in main thread).
        self.num_threads = 0
        
        # Batch size. Default is 32, which works well in most cases.
        self.batch_size = 32
        
        # Maximum number of allowed samples. Default is inf.
        self.max_dataset_size = 'inf'
        
        # Type of preprocessing input parameters undergo: add_noise or none. Default is none.
        self.preprocess = 'none'
        
        # Amount of noise if add_noise preprocessing is selected. Default is 0.005
        self.noise_factor = 0.005

        ###########################
        ## Additional parameters ##
        ###########################
        # Frequency of showing training loss and info on UE console.
        self.print_freq = 100
