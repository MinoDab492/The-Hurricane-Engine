# -*- coding: utf-8 -*-
"""
Copyright Epic Games, Inc. All Rights Reserved.
"""

import os
import time
import unreal as ue
import numpy as np
import torch
import torch.nn as nn
import datetime
from torch.utils.data import DataLoader


class TensorUploadDataset:
    def __init__(self, inputs, outputs, device):
        self.inputs = inputs
        self.outputs = outputs
        self.device = device

    def __getitem__(self, index):
        return (self.inputs[index], self.outputs[index])

    def __len__(self):
        return len(self.inputs)

    def collate(self, args):
        return (
            torch.tensor(np.concatenate([a[0][None] for a in args], axis=0), device=self.device),
            torch.tensor(np.concatenate([a[1][None] for a in args], axis=0), device=self.device))


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, x):
        return (x - self.mean) / self.std


class Denormalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, x):
        return x * self.std + self.mean


def create_network(
    input_size,
    output_size,
    hidden_size,
    num_layers,
    input_mean,
    input_std,
    output_mean,
    output_std):
    network = nn.Sequential()
    network.add_module('normalize', Normalize(input_mean, input_std))

    for i in range(num_layers + 1):
        if i == 0:
            network.add_module('linear_%i' % i, nn.Linear(input_size, hidden_size))
            network.add_module('activation_%i' % i, nn.ELU())
        elif i == num_layers:
            network.add_module('linear_%i' % i, nn.Linear(hidden_size, output_size))
        else:
            network.add_module('linear_%i' % i, nn.Linear(hidden_size, hidden_size))
            network.add_module('activation_%i' % i, nn.ELU())

    network.add_module('denormalize', Denormalize(output_mean, output_std))
    return network


def generate_time_strings(cur_iteration, total_num_iterations, start_time):
    iterations_remaining = total_num_iterations - cur_iteration
    passed_time = time.time() - start_time
    avg_iteration_time = passed_time / (cur_iteration + 1)
    est_time_remaining = iterations_remaining * avg_iteration_time
    est_time_remaining_string = str(datetime.timedelta(seconds=int(est_time_remaining)))
    passed_time_string = str(datetime.timedelta(seconds=int(passed_time)))
    return passed_time_string, est_time_remaining_string


def train(self):
    # Training Parameters
    training_state = 'none'
    seed = 1234
    batch_size = self.get_model().batch_size
    niter = self.get_model().num_iterations
    lr = self.get_model().learning_rate
    lr_gamma = 0.99

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
        torch.set_num_threads(1)

    np.random.seed(seed)
    torch.manual_seed(seed)

    # Make Training Dir
    checkpoints_dir = ue.Paths.convert_relative_path_to_full(ue.Paths.project_intermediate_dir())
    training_dir = os.path.join(checkpoints_dir, 'VertexDeltaModel')

    # Remove any existing files
    if os.path.exists(training_dir):
        for f in os.listdir(training_dir):
            if os.path.isfile(os.path.join(training_dir, f)):
                os.remove(os.path.join(training_dir, f))

    # Recreate directory
    if not os.path.exists(training_dir):
        os.makedirs(training_dir)

    # Dataset
    num_samples = self.num_samples()
    num_bone_values = 6 * self.get_number_sample_transforms()  # 2 columns of 3x3 rotation matrix per bone, so 2x3
    num_curve_values = self.get_number_sample_curves()
    num_delta_values = 3 * self.get_number_sample_deltas()  # xyz per vertex delta

    # Create inputs/outputs. Memory map outputs array as it can be very large
    inputs = np.empty([num_samples, num_bone_values + num_curve_values], dtype=np.float32)
    outputs = np.memmap(os.path.join(training_dir, 'outputs.bin'), dtype=np.float32, mode='w+',
                        shape=(num_samples, num_delta_values))

    # Precompute all inputs/outputs
    data_start_time = time.time()
    try:
        with ue.ScopedSlowTask(num_samples, "Sampling Frames") as sampling_task:
            sampling_task.make_dialog(True)
            for i in range(num_samples):
                # Stop if the user has pressed Cancel in the UI
                if sampling_task.should_cancel():
                    raise GeneratorExit('CannotUse')

                # Set the sample
                sample_exists = self.set_current_sample_index(int(i))
                assert (sample_exists)

                # Copy inputs
                inputs[i, :num_bone_values] = self.sample_bone_rotations
                inputs[i, num_bone_values:] = self.sample_curve_values

                # Copy outputs
                outputs[i] = self.sample_deltas

                # Calculate passed and estimated time and report progress
                passed_time_string, est_time_remaining_string = generate_time_strings(i, num_samples, data_start_time)
                sampling_task.enter_progress_frame(1,
                                                   f'Sampling frame {i + 1:6d} of {num_samples:6d} - Time: {passed_time_string} - Left: {est_time_remaining_string}')

    except GeneratorExit as message:
        ue.log_warning("Sampling frames canceled by user.")
        if str(message) == 'CannotUse':
            return 2  # 'aborted_cant_use'
        else:
            return 1  # 'aborted'

    data_elapsed_time = time.time() - data_start_time
    ue.log(f'Calculating inputs and outputs took {data_elapsed_time:.0f} seconds.')

    # Now dataset is constructed, reload memory mapped file in read-only mode
    outputs = np.memmap(os.path.join(training_dir, 'outputs.bin'), dtype=np.float32, mode='r',
                        shape=(num_samples, num_delta_values))

    # Make Dataset and DataLoader
    dataset = TensorUploadDataset(inputs, outputs, device)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=dataset.collate)

    # Input uses simple mean
    input_mean = inputs.mean(axis=0)

    # Input std is averaged for each type of input (bone rotations / curves)
    input_std = np.ones_like(input_mean)
    if num_bone_values > 0: input_std[:num_bone_values] = inputs[:, :num_bone_values].std(axis=0).mean() + 1e-5
    if num_curve_values > 0: input_std[num_bone_values:] = inputs[:, num_bone_values:].std(axis=0).mean() + 1e-5

    # Output uses simple mean and std
    output_mean = outputs.mean(axis=0)
    output_std = outputs.std(axis=0)

    # Upload mean and std tensors
    input_mean = torch.tensor(input_mean, device=device)
    input_std = torch.tensor(input_std, device=device)
    output_mean = torch.tensor(output_mean, device=device)
    output_std = torch.tensor(output_std, device=device)

    # Create Network
    hidden_size = self.get_model().num_neurons_per_layer
    num_layers = self.get_model().num_hidden_layers

    network = create_network(
        num_bone_values + num_curve_values, num_delta_values, hidden_size, num_layers,
        input_mean, input_std,
        output_mean, output_std)

    network.to(device)

    # Optimizer and Scheduler
    optimizer = torch.optim.AdamW(
        list(network.parameters()),
        lr=lr,
        amsgrad=True)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_gamma)

    # Train    
    try:
        training_start_time = time.time()
        with ue.ScopedSlowTask(niter, "Training Model") as training_task:
            training_task.make_dialog(True)
            rolling_loss = []

            for i in range(niter):
                # Stop if the user has pressed Cancel in the UI
                if training_task.should_cancel():
                    raise GeneratorExit()

                # Zero Grad
                optimizer.zero_grad()

                # Sample Data
                X, Y = next(iter(dataloader))

                # Compute Loss                
                loss = torch.mean(torch.abs(Y - network(X)))
                loss.backward()

                # Optimize
                optimizer.step()

                # Get the loss
                loss_item = loss.item()
                rolling_loss.append(loss_item)
                rolling_loss = rolling_loss[-1000:]

                # Decay lr every 1000 iterations
                if i % 1000 == 0:
                    scheduler.step()

                # Calculate passed and estimated remaining time, and report progress
                passed_time_string, est_time_remaining_string = generate_time_strings(i, niter, training_start_time)
                training_task.enter_progress_frame(
                    1,
                    f'Training iteration: {i + 1:6d} of {niter:6d} - Time: {passed_time_string} - Left: {est_time_remaining_string} - Avg error: {np.mean(rolling_loss):.5f} cm')

        ue.log("Model successfully trained.")
        return 0  # 'succeeded'

    except GeneratorExit as message:
        ue.log_warning("Training canceled by user.")
        return 1  # 'aborted'

    finally:
        # Save Final Version
        X, _ = next(iter(dataloader))
        if os.name != "posix":
            torch.onnx.export(
                network, X[:1],
                os.path.join(training_dir, 'VertexDeltaModel.onnx'),
                verbose=False,
                export_params=True,
                input_names=['InputParams'],
                output_names=['OutputPredictions'])
            training_state = 'succeeded'
        else:
            ue.log_warning('Pytorch on linux does not support onnx export.')
            training_state = 'aborted'
            model_path = os.path.join(training_dir, 'VertexDeltaModel.pt')
            model_dict_path = os.path.join(training_dir, 'VertexDeltaModel_StateDict.pth')
            torch.save(network, model_path)
            torch.save(network.state_dict(), model_dict_path)

        # Remove memory mapped data
        outputs._mmap.close()
        if os.path.exists(os.path.join(training_dir, 'outputs.bin')):
            os.remove(os.path.join(training_dir, 'outputs.bin'))
