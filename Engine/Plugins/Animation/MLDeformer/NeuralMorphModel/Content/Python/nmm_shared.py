# -*- coding: utf-8 -*-
'''
Copyright Epic Games, Inc. All Rights Reserved.
'''
import time
import datetime
import json
import unreal as ue
import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

'''
Import tensorboard related things.

In order for this to work, do the following things.
First install anaconda, which you can download at https://www.anaconda.com/

Then inside the anaconda command prompt create some environment and install tensorboard.
In the example below we create an environment named 'unreal', but this name can be anything.
Try to use the same python version as Unreal uses.

    conda create -n unreal python=3.9.7
    activate unreal
    conda install tensorboard

After doing this, launch python and import sys and show the system path.

    python
    >>import sys
    >>sys.path

This will show a few path strings. One of them, contains the site-packages folder, copy that one.
Now replace the path in the sys.path.append in the code below, with the one you just copied.

Also make sure to set the dev_mode variable to True in order to use Tensorboard.

In order to view the Tensorboard after training, go to the (YourProjectDir)/Intermediate/NeuralMorphModel/ folder
inside the anaconda command prompt. After that launch tensorboard and then navigate to the reported url in your browser. 

So something like:

    cd MyProject/Intermediate/NeuralMorphModel
    tensorboard --logdir TensorBoard

Then inside your browser, go to the URL that tensorboard reports, for example:

    http://localhost:6006

That should show you the tensorboard for your model.
Also make sure to delete the TensorBoard sub-folder inside the intermediate folder in case you want to clear the 
history of all results.
'''

dev_mode = False
if dev_mode:
    packages_path = 'C:\\Users\\JohnvanderBurg\\anaconda3\\envs\\unreal\\lib\\site-packages'
    if packages_path not in sys.path:
        sys.path.append(packages_path)
    from torch.utils.tensorboard import SummaryWriter


class TensorUploadDataset(Dataset):
    def __init__(self, inputs, outputs, device):
        self.inputs = inputs
        self.outputs = outputs
        self.device = device

    def __getitem__(self, index):
        return self.inputs[index], self.outputs[index]

    def __len__(self):
        return len(self.inputs)

    def collate(self, args):
        return (torch.tensor(np.concatenate([a[0][None] for a in args], axis=0), device=self.device),
                torch.tensor(np.concatenate([a[1][None] for a in args], axis=0), device=self.device))


def train(training_model, create_network_function, include_curves):
    training_state = 'none'

    # Make intermediate training dir.
    checkpoints_dir = ue.Paths.convert_relative_path_to_full(ue.Paths.project_intermediate_dir())
    training_dir = os.path.join(checkpoints_dir, 'NeuralMorphModel')
    inputs_filename, outputs_filename = get_inputs_outputs_filename(training_dir)

    # Remove any existing files/folders in the intermediate folder for this model.
    # Do not delete the files in the ignore list though.
    ignore_list = list()
    ignore_list.append(inputs_filename)
    ignore_list.append(outputs_filename)
    ignore_list.append(get_runs_filename())
    if os.path.exists(training_dir):
        for f in os.listdir(training_dir):
            filename = os.path.join(training_dir, f)
            if filename not in ignore_list:
                if os.path.isfile(filename):
                    os.remove(filename)

    # If our intermediate training folder doesn't exist yet, create it.
    if not os.path.exists(training_dir):
        os.makedirs(training_dir)

    # Get the next run index, which is some persistent index, used to generate unique names for our training runs.
    run_index = get_next_run_index()

    seed = 777
    model = training_model.get_model()
    batch_size = model.batch_size
    num_iters = model.num_iterations
    lr = model.learning_rate
    regularization_factor = model.regularization_factor

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    ue.log(f'Training using device: {device}')

    np.random.seed(seed)
    torch.manual_seed(seed)

    # Get some numbers for the inputs and outputs.
    num_bones = training_model.get_number_sample_transforms()
    num_samples = training_model.num_samples()
    num_bone_values = 6 * num_bones  # 2 columns of 3x3 rotation matrix per bone, so 3x2.
    num_curve_values = training_model.get_number_sample_curves() if include_curves else 0
    num_vertices = training_model.get_number_sample_deltas()
    num_delta_values = 3 * num_vertices  # xyz per vertex delta.
    num_inputs = num_bone_values + num_curve_values

    # Create the tensorboard summary writer in the intermediate folder.
    if dev_mode:
        tensorboard_writer = init_tensorboard(training_model, training_dir, run_index)
    else:
        tensorboard_writer = None

    # Compute all inputs/outputs if needed.
    try:
        # If the input and output files don't exist, or the sizes don't match or our
        # input settings to the network changed, then regenerate.
        input_output_files_exist = os.path.exists(inputs_filename) and os.path.exists(outputs_filename)
        if (not input_output_files_exist) or training_model.get_needs_resampling():
            sampling_start_time = time.time()
            generate_inputs_outputs(
                training_model,
                inputs_filename,
                outputs_filename,
                sampling_start_time,
                include_curves)
            data_elapsed_time = time.time() - sampling_start_time
            ue.log(f'Calculating inputs and outputs took {data_elapsed_time:.0f} seconds.')

    except GeneratorExit as message:
        ue.log_warning('Sampling frames canceled by user.')
        if str(message) == 'CannotUse':
            training_state = 'aborted_cant_use'
            return 2  # 'aborted_cant_use'
        else:
            training_state = 'aborted'
            return 1  # 'aborted'

    # Now dataset is constructed, reload memory mapped files in read-only mode.
    inputs = np.memmap(inputs_filename, dtype=np.float32, mode='r', shape=(num_samples, num_inputs))
    outputs = np.memmap(outputs_filename, dtype=np.float32, mode='r', shape=(num_samples, num_delta_values))

    # Create the dataset and data loader.
    dataset = TensorUploadDataset(inputs, outputs, device)

    # Split into train and test set.
    if dev_mode:
        test_dataset_percentage = 0.2  # 20% test set, 80% training set.
        train_size = int((1.0 - test_dataset_percentage) * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        ue.log(f'Split dataset of {len(dataset)} items into a training set of {len(train_dataset)} ' +
               f'and test set of {len(test_dataset)} items')
    else:
        test_dataset = None
        train_dataset = dataset

    # Create the data loader on the training set.
    dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=dataset.collate)

    # Create the data loader on the test set.
    if dev_mode:
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=dataset.collate)
    else:
        test_dataloader = None

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

    # We don't use shared joints for now.
    shared_joints = None

    # Create the neural network.
    hidden_size = model.local_num_neurons_per_layer if model.mode == 0 else model.global_num_neurons_per_layer
    num_layers = model.local_num_hidden_layers if model.mode == 0 else model.global_num_hidden_layers
    num_morph_targets = model.local_num_morph_targets_per_bone if model.mode == 0 else model.global_num_morph_targets
    network = create_network_function(
        num_vertices,
        num_inputs,
        num_morph_targets,
        num_bones,
        num_layers,
        hidden_size,
        shared_joints,
        device,
        input_mean=input_mean,
        input_std=input_std,
        output_mean=output_mean,
        output_std=output_std
    ).to(device)

    # Create the optimizer and scheduler.
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    lr_decay = model.learning_rate_decay
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=lr_decay)

    # Network training loop.    
    try:
        training_start_time = time.time()
        with ue.ScopedSlowTask(num_iters, 'Training Model') as training_task:
            training_task.make_dialog(True)
            rolling_loss = []

            for i in range(num_iters):
                # Stop if the user has pressed Cancel in the UI.
                if training_task.should_cancel():
                    raise GeneratorExit()

                # Set all tensor gradients to zero.
                optimizer.zero_grad()

                # Get a batch of training data.
                x, y = next(iter(dataloader))

                # Compute the loss.
                loss = torch.mean(torch.abs(y - network.forward(x)))
                error_in_cm = loss.item()

                # L1 Regularization.
                if regularization_factor > 0.0:
                    loss += torch.mean(torch.abs(network.morph_target_matrix)) * regularization_factor

                loss.backward()
                optimizer.step()

                # Get the loss.
                rolling_loss.append(error_in_cm)
                rolling_loss = rolling_loss[-1000:]

                # Get some other values.
                avg_error_cm = np.mean(rolling_loss)
                current_lr = scheduler.get_last_lr()[0]

                # Write tensorboard scalars.
                if dev_mode:
                    # Evaluate test set.
                    if i % 5 == 0:
                        x_test, y_test = next(iter(test_dataloader))
                        test_loss = torch.mean(torch.abs(y_test - network.forward(x_test)))
                        if tensorboard_writer:
                            tensorboard_writer.add_scalar('loss/test', test_loss, i + 1)

                if tensorboard_writer:
                    tensorboard_writer.add_scalar('loss/train', loss, i + 1)
                    tensorboard_writer.add_scalar('lr/train', current_lr, i + 1)

                if torch.cuda.is_available() and device != 'cpu':
                    cuda_allocated_gb = torch.cuda.memory_allocated(device) / (1024 * 1024 * 1024)

                # Decay the learning rate.
                scheduler.step()

                # Calculate passed and estimated remaining time, and report progress.
                passed_time_string, est_time_remaining_string = generate_time_strings(i, num_iters, training_start_time)
                progress_string = \
                    f'Training iteration: {i + 1:6d} of {num_iters} - ' + \
                    f'Time: {passed_time_string} - Left: {est_time_remaining_string} - '
                if torch.cuda.is_available() and device != 'cpu':
                    progress_string += f'GPU: {cuda_allocated_gb:.2f} gb - '
                progress_string += f'Avg error: {avg_error_cm:.5f} cm'
                training_task.enter_progress_frame(1, progress_string)
                if i % 100 == 0 or i == num_iters - 1:
                    ue.log(progress_string + f' - lr: {current_lr:.5f}')
            ue.log('Model successfully trained.')
            training_state = 'succeeded'
            return 0  # 'succeeded'

    except GeneratorExit as message:
        ue.log_warning('Training canceled by user.')
        training_state = 'aborted'
        return 1  # 'aborted'

    finally:
        # Extract the morph targets.
        extract_morph_targets(network, training_model, output_mean, output_std)

        # Save final version.
        x, _ = next(iter(dataloader))

        # Close the tensorboard writer. 
        if dev_mode:
            tensorboard_writer.close()

        # Save the onnx file, which our ML Deformer model will load.
        network.eval()
        if os.name != "posix":
            torch.onnx.export(
                network, x[:1],
                os.path.join(training_dir, 'NeuralMorphModel.onnx'),
                verbose=False,
                export_params=True,
                input_names=['InputParams'],
                output_names=['OutputPredictions'])
        else:
            ue.log_warning('Pytorch on linux does not support onnx export.')
            training_state = 'aborted'
            model_path = os.path.join(training_dir, 'NeuralMorphModel.pt')
            model_dict_path = os.path.join(training_dir, 'NeuralMorphModel_StateDict.pth')
            torch.save(network, model_path)
            torch.save(network.state_dict(), model_dict_path)

        # Remove memory mapped files.
        outputs._mmap.close()
        inputs._mmap.close()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# Generate the memory mapped files with input and output data.
def generate_inputs_outputs(training_model, inputs_filename, outputs_filename, data_start_time, include_curves):
    num_samples = training_model.num_samples()
    num_bone_values = 6 * training_model.get_number_sample_transforms()  # 2 columns of 3x3 rotation matrix per bone.
    num_curve_values = training_model.get_number_sample_curves() if include_curves else 0
    num_delta_values = 3 * training_model.get_number_sample_deltas()  # xyz per vertex delta.

    # Create inputs/outputs. Memory map inputs and outputs.
    num_inputs = num_bone_values + num_curve_values
    inputs = np.memmap(inputs_filename, dtype=np.float32, mode='w+', shape=(num_samples, num_inputs))
    outputs = np.memmap(outputs_filename, dtype=np.float32, mode='w+', shape=(num_samples, num_delta_values))

    # Iterate over all sample frames.
    with ue.ScopedSlowTask(num_samples, 'Sampling Frames') as sampling_task:
        sampling_task.make_dialog(True)
        for i in range(num_samples):
            # Stop if the user has pressed Cancel in the UI.
            # The 'CannotUse' makes sure the user doesn't get offered to use the partially trained network, because
            # at this stage we didn't even create and train a network yet.
            if sampling_task.should_cancel():
                raise GeneratorExit('CannotUse')

            # Set the sample.
            sample_exists = training_model.set_current_sample_index(int(i))
            assert sample_exists

            # Copy inputs.
            inputs[i, :num_bone_values] = training_model.sample_bone_rotations
            if include_curves:
                inputs[i, num_bone_values:] = training_model.sample_curve_values

            # Copy outputs
            outputs[i] = training_model.sample_deltas

            # Calculate passed and estimated time and report progress.
            passed_time_string, est_time_remaining_string = generate_time_strings(i, num_samples, data_start_time)
            sampling_task.enter_progress_frame(
                1,
                f'Sampling frame {i + 1:6d} of {num_samples} - '
                f'Time: {passed_time_string} - Left: {est_time_remaining_string}')

    # Save the generated inputs and outputs.
    inputs._mmap.close()
    outputs._mmap.close()


# Extract the morph targets from the neural network.
def extract_morph_targets(network, training_model, output_mean, output_std):
    with ue.ScopedSlowTask(2, 'Extracting Morph Targets') as morph_task:
        morph_task.make_dialog(True)

        # Pre-multiply the morph target deltas with the standard deviation.
        morph_target_matrix = network.morph_target_matrix * output_std.unsqueeze(dim=-1)
        morph_task.enter_progress_frame(1)
        num_morph_targets = morph_target_matrix.shape[-1] + 1  # Add one, because we add the means as well.
        print('Num Morph Targets: ', num_morph_targets)

        # Store the means as first morph target, then add the generated morphs to it.
        deltas = output_mean.cpu().detach().numpy().tolist()
        deltas.extend(morph_target_matrix.T.flatten().cpu().detach().numpy().tolist())
        training_model.get_model().set_morph_target_delta_floats(deltas)

        morph_task.enter_progress_frame(1)


# Generate the elapsed and remaining time strings that we show in the progress window. 
def generate_time_strings(cur_iteration, total_num_iterations, start_time):
    iterations_remaining = total_num_iterations - cur_iteration
    passed_time = time.time() - start_time
    avg_iteration_time = passed_time / (cur_iteration + 1)
    est_time_remaining = iterations_remaining * avg_iteration_time
    est_time_remaining_string = str(datetime.timedelta(seconds=int(est_time_remaining)))
    passed_time_string = str(datetime.timedelta(seconds=int(passed_time)))
    return passed_time_string, est_time_remaining_string


# Get the filename that keeps track of information about runs.
def get_runs_filename():
    runs_filename = ue.Paths.convert_relative_path_to_full(ue.Paths.project_intermediate_dir())
    runs_filename = os.path.join(runs_filename, 'NeuralMorphModel')
    runs_filename = os.path.join(runs_filename, 'runs.json')
    return runs_filename


# Save a dictionary to the runs json file.
def save_runs_json(dict_to_write):
    try:
        with open(get_runs_filename(), 'w') as outfile:
            json.dump(dict_to_write, outfile, indent=4, sort_keys=True)
    except IOError as e:
        pass


# Read the runs json file.
def read_runs_json():
    try:
        with open(get_runs_filename()) as runs_file:
            runs = json.load(runs_file)
            return runs
    except IOError as e:
        pass
    runs = dict()
    runs['LastRunIndex'] = 0
    return runs


# Get the filenames for the inputs and outputs files.
def get_inputs_outputs_filename(training_dir):
    inputs_filename = os.path.join(training_dir, 'inputs.bin')
    outputs_filename = os.path.join(training_dir, 'outputs.bin')
    return inputs_filename, outputs_filename


# Load the runs dictionary. This gives us a run counter that is increased each time.
# It is mainly used for Tensorboard support, where we give each run a unique name.
def get_next_run_index():
    read_runs_json()
    runs_dict = read_runs_json()
    run_index = int(runs_dict['LastRunIndex']) + 1 if 'LastRunIndex' in runs_dict else 1
    runs_dict['LastRunIndex'] = run_index
    save_runs_json(runs_dict)
    return run_index


# Initialize the Tensorboard SummaryWriter. 
def init_tensorboard(training_model, training_dir, run_index):
    num_samples = training_model.num_samples()
    num_iters = training_model.get_model().num_iterations
    batch_size = training_model.get_model().batch_size
    lr = training_model.get_model().learning_rate
    num_morph_targets_per_bone = training_model.get_model().local_num_morph_targets_per_bone
    regularization = training_model.get_model().regularization_factor
    tensorboard_dir = os.path.join(training_dir, 'TensorBoard')
    tensorboard_dir = ue.Paths.convert_relative_path_to_full(tensorboard_dir)
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    model = training_model.get_model()
    lr_string = f'{lr:.6f}'.rstrip('0').rstrip('.')
    lr_string = lr_string.split('.')[1]
    reg_string = f'{regularization:.6f}'.rstrip('0').rstrip('.')
    mode_string = 'local' if model.mode == 0 else 'global'
    morphs_per_bone_string = f'{num_morph_targets_per_bone}mtpb_' if model.mode == 0 else ''
    num_transforms = training_model.get_number_sample_transforms()
    num_local_morphs = num_transforms * model.local_num_morph_targets_per_bone
    total_morphs = num_local_morphs if model.mode == 0 else model.global_num_morph_targets
    num_hidden_layers = model.local_num_hidden_layers if model.mode == 0 else model.global_num_hidden_layers
    num_neurons_per_layer = model.local_num_neurons_per_layer if model.mode == 0 else model.global_num_neurons_per_layer
    session_name = \
        f'{model.get_outer().get_name()}_' + \
        f'run{run_index:03d}_' + \
        f'{mode_string}_' + \
        f'{total_morphs}mt_' + \
        f'{morphs_per_bone_string}' + \
        f'{num_samples}s_' + \
        f'{num_hidden_layers}l_' + \
        f'{num_neurons_per_layer}u_' + \
        f'{num_iters}it_' + \
        f'{batch_size}b_' + \
        f'{lr_string}lr_' + \
        f'{reg_string}reg'
    tensorboard_dir = tensorboard_dir + f'/{session_name}'

    tensorboard_writer = SummaryWriter(log_dir=tensorboard_dir, flush_secs=5)
    session_description = \
        f'Asset: {model.get_outer().get_name()}<br>' + \
        f'Run: {run_index}<br>' + \
        f'Mode: {mode_string}<br>' + \
        f'Morph Targets: {total_morphs}<br>' + \
        f'Iterations: {num_iters}<br>' + \
        f'Num Samples: {num_samples}<br>' + \
        f'Num Hidden Layers: {num_hidden_layers}<br>' + \
        f'Num Units Per Layer: {num_neurons_per_layer}<br>' + \
        f'Batch Size: {batch_size}<br>' + \
        f'Learning Rate: {lr:.6f}<br>' + \
        f'Regularization Factor: {regularization:.6f}<br>' + \
        f'Num Transforms: {training_model.get_number_sample_transforms()}<br>' + \
        f'Num Curves: {training_model.get_number_sample_curves()}'
    tensorboard_writer.add_text('description', session_description)
    return tensorboard_writer
