# -*- coding: utf-8 -*-
"""
MLDeformerTrainingModel class.

Copyright Epic Games, Inc. All Rights Reserved.
"""

import os
import torch
import traceback
import unreal
from unreal import Paths

from mldeformer.samples.training.train import train_network
from mldeformer.training.options.train_options import TrainOptions
from mldeformer.utils.io import os_utils


def unload():
    """Unload every module
    This enables re-import of this package without restarting the
    interpreter, whilst also accounting for import order to avoid/bypass
    cyclical dependencies.
    """   
    import sys  # Local import, to prevent leakage
    for key, _ in sys.modules.copy().items():
        if key.startswith(__name__):
            sys.modules.pop(key)


@unreal.uclass()
class MLDeformerTrainingModel(unreal.MLDeformerPythonTrainingModel):
    @unreal.ufunction(override=True)
    def train(self):
        # Get default training options.
        train_options = TrainOptions()
        try:
            # Customize training options.
            deformer_asset = self.get_deformer_asset()
            train_options = self._customize_training(
                train_options, deformer_asset, self.data_set_interface)
            train_options.print_options()

            # Train network with custom training options and asset data.
            train_network(self.data_set_interface, train_options)
            unreal.log("Model successfully trained.")
            return 0  # 'succeeded'
        except GeneratorExit as message:
            if str(message) != 'CannotUse':
                unreal.log_warning("Training canceled by user.")
                return 1  # 'aborted'
            else:
                unreal.log_warning("Training canceled by user, cannot use network.")
                return 2  # 'aborted_cant_use'
        except StopIteration as si:
            unreal.log_warning("Data loader failed: {}".format(si))
            return 3  # 'data_loader_failed'
        except Exception as e:
            unreal.log_warning("Training failed: Unexpected error. \n {}".format(e))
            traceback.print_exc()
            return 4  # 'unknown_error'
        finally:
            if train_options.gpu_ids and torch.cuda.is_available():
                torch.cuda.empty_cache()
                
    def _customize_training(self, train_opts, ml_deformer_asset, data_set_interface):
        # Architecture parameters.
        train_opts.input_nc = 4 * data_set_interface.get_number_sample_transforms() + \
                              data_set_interface.get_number_sample_curves()
        train_opts.output_nc = 3 * data_set_interface.get_number_sample_deltas()
        train_opts.n_linear_layers_G = ml_deformer_asset.num_hidden_layers
        train_opts.netG = '{}_{}'.format(train_opts.model, ml_deformer_asset.num_neurons_per_layer)
        train_opts.batch_size = ml_deformer_asset.batch_size

        # Activation function.
        train_opts.activation = 'relu'
        if ml_deformer_asset.activation_function == 1:
            train_opts.activation = 'lrelu'
        elif ml_deformer_asset.activation_function == 2:
            train_opts.activation = 'tanh'

        # Loss function.
        train_opts.loss_type = 'L1'
        if ml_deformer_asset.loss_function == 1:
            train_opts.loss_type = 'MSE'
        elif ml_deformer_asset.loss_function == 2:
            train_opts.loss_type = 'Shrinkage'
            train_opts.shrink_speed = ml_deformer_asset.shrinkage_speed
            train_opts.shrink_loc = ml_deformer_asset.shrinkage_threshold

        # Normal and decay iteration to adapt learning rate.
        train_opts.niter = ml_deformer_asset.epochs
        train_opts.niter_decay = ml_deformer_asset.epochs_with_decay

        # Initial learning rate.
        train_opts.lr = ml_deformer_asset.learning_rate

        # Learning rate policy.
        train_opts.lr_policy = 'linear'
        if ml_deformer_asset.decay_function == 1:
            train_opts.lr_policy = 'multiplicative'
        train_opts.lr_gamma = ml_deformer_asset.decay_rate

        # Preprocessing function.
        train_opts.preprocess = 'add_noise'
        if ml_deformer_asset.noise_amount > 0.0:
            train_opts.noise_factor = ml_deformer_asset.noise_amount / 100.0
        else:
            train_opts.preprocess = 'none'
            train_opts.noise_factor = 0.0

        if ml_deformer_asset.device_type == 1:
            # Find available GPU id.
            if torch.cuda.is_available():
                current_device = torch.cuda.current_device()
                train_opts.gpu_ids = [current_device]
                torch.cuda.set_device(current_device)
            else:
                train_opts.gpu_ids = []

        # Caching Options
        train_opts.memory_cache_in_megabytes = ml_deformer_asset.cache_size_in_megabytes
        
        train_opts.checkpoints_dir = Paths.convert_relative_path_to_full(Paths.project_intermediate_dir())
        training_dir = os.path.join(train_opts.checkpoints_dir, train_opts.name)
        os_utils.rmfiles(training_dir)
        os_utils.mkdir(training_dir)

        train_opts.max_dataset_size = data_set_interface.num_samples()

        return train_opts
