# -*- coding: utf-8 -*-
"""
General-purpose training script for parameter-to-delta deformation.

This script works for various custom models (with option 'model': e.g., deep_deform) and
different datasets (with option 'dataset_mode': e.g., unreal_deformer).

First, it creates a model, dataset, and visualizer given the user-defined options.
Then, it trains the network. During the training, this script logs the loss and save models.
The script supports continue/resume training. Use 'continue_train' to resume your previous training.
This optional but not used at the moment.

See training/options/base_options.py and training/options/train_options.py for further information on training options.

Copyright Epic Games, Inc. All Rights Reserved.
"""

import os
import sys
import unreal
import datetime

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, root_dir)
from mldeformer.training.datasets import create_dataset
from mldeformer.training.models import create_model
from mldeformer.training.logs.model_visualizer import Visualizer
from mldeformer.utils.misc.timer import Timer


def train_network(data_interface, train_opts):
    dataset_timer = Timer('s')
    dataset_timer.start()
    dataset = create_dataset(data_interface, train_opts)
    print('Number of training samples = %d' % len(dataset))
    dataset_timer.stop()
    print('Dataset took %f seconds to prepare' % dataset_timer.show())

    # Create a model based on train_opts.model and other options.
    model = create_model(train_opts)

    # Load, print networks, and create schedulers.
    model.setup(train_opts)

    # Create visualizer to display loss info and stats.
    visualizer = Visualizer(train_opts)

    # Initialize timers.
    batch_optim_timer = Timer('s')
    epoch_timer = Timer('s')
    training_timer = Timer('s')
    training_timer.start()
    total_iters = 0
    estimated_time_left = 0.0
    expected_iters = (train_opts.niter + train_opts.niter_decay) * (
        (len(dataset) // train_opts.batch_size) + 1) * train_opts.batch_size
    range_min = train_opts.epoch_count
    range_max = train_opts.niter + train_opts.niter_decay + 1
    total_epochs = range_max - range_min
    average_epoch_time = 0.0
    loss_value = 0.0

    with unreal.ScopedSlowTask(expected_iters, "Training Model") as training_task:
        training_task.make_dialog(True)

        # Outer loop for different epochs. The saved model is identified by <epoch_count>+<save_epoch_freq>
        for epoch in range(train_opts.epoch_count, train_opts.niter + train_opts.niter_decay + 1):
            epoch_iter = 0
            epoch_timer.start()

            # Inner loop within one epoch.
            for i, data in enumerate(dataset):
                if training_task.should_cancel():  # True if the user has pressed Cancel in the UI
                    model.convert()
                    model.save_converted_networks('latest')
                    raise GeneratorExit()

                # Reset visualizer at each iteration.
                visualizer.reset()
                total_iters += train_opts.batch_size
                epoch_iter += train_opts.batch_size

                # Unpack data from dataset and apply preprocessing.
                model.set_input(data)

                # Calculate loss functions, get gradients, update network weights.
                batch_optim_timer.start()
                model.optimize_parameters()
                batch_optim_timer.stop()
                batch_optim_time = batch_optim_timer.show()

                # Get the current loss value.
                losses = model.get_current_losses()
                loss_list = list(losses.values())
                loss_value = loss_list[-1] if loss_list else 0.0

                # Print training losses to console.
                if total_iters % train_opts.print_freq == 0:
                    visualizer.print_current_losses(epoch, epoch_iter, losses, batch_optim_time)

                # Calculate some time statistics.
                training_time_passed = training_timer.time_passed()
                time_string = str(datetime.timedelta(seconds=int(training_time_passed)))
                epochs_remaining = total_epochs - (epoch - 1)

                if epoch > 1:
                    estimated_time_left = epochs_remaining * average_epoch_time
                    estimated_time_left_string = str(datetime.timedelta(seconds=int(estimated_time_left)))
                else:
                    estimated_time_left_string = '<wait>'

                # Report our progress and current state to the task window.
                progress_message = f'Epoch: {epoch}, Time: {time_string}, Left: {estimated_time_left_string}, Loss: {loss_value:.5f}'
                training_task.enter_progress_frame(train_opts.batch_size, progress_message)

            # Cache our model every <save_epoch_freq> epochs.
            if train_opts.save_epochs and epoch % train_opts.save_epoch_freq == 0:
                print('Saving the model at the end of epoch {:d}, iters {:d}'.format(epoch, total_iters))
                model.convert()
                model.save_converted_networks(epoch)
                model.save_converted_networks('latest')

            epoch_timer.stop()
            print('End of epoch {:d} / {:d} \t Time Taken: {:f} sec'.format(epoch,
                                                                            train_opts.niter + train_opts.niter_decay,
                                                                            epoch_timer.show()))

            average_epoch_time = training_timer.time_passed() / float(epoch)

            # Update learning rates at the end of every epoch.
            model.update_learning_rate()

    model.convert()
    model.save_converted_networks('latest')
