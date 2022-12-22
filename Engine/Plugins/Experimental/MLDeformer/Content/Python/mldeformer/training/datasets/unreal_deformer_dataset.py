# -*- coding: utf-8 -*-
"""
ParametersPointsDataset class.

Copyright Epic Games, Inc. All Rights Reserved.
"""

import torch
import unreal

from mldeformer.training.transformers.data_transformer import get_params_1d, get_transform_1d
from mldeformer.utils.misc.timer import Timer
from .base_dataset import BaseDataset

class BaseUEDeformerDataset(BaseDataset):
    """ Common Deformer Dataset functionality"""

    def __init__(self, ml_deformer_dataset, opt):
        """ Initialize this dataset class. """
        BaseDataset.__init__(self, opt)
        # Initialize parameters.
        self.ml_deformer_dataset = ml_deformer_dataset
        self.has_normalization_params = True
        self.vertex_delta_mean_cpu = torch.tensor([0.0, 0.0, 0.0])
        self.vertex_delta_scale_cpu = torch.tensor([1.0, 1.0, 1.0])
        # Compute statistics.
        self.is_initialized = self._update_delta_statistics()
        self.precision = opt.precision

    def _update_delta_statistics(self):
        """" Update vertex delta mean and scale. """
        timer = Timer('s')
        timer.start()
        if not self.ml_deformer_dataset.compute_deltas_statistics():
            raise GeneratorExit('CannotUse')
    
        timer.stop()
        print('Computing vertex delta statistics took {} seconds'.format(timer.show()))
    
        # Get vertex delta mean and scale.
        delta_mean = self.ml_deformer_dataset.vertex_delta_mean
        delta_scale = self.ml_deformer_dataset.vertex_delta_scale
        ones = unreal.Vector(1.0, 1.0, 1.0)
        if delta_mean.length() < 1e-5 and (delta_scale - ones).length() < 1e-5:
            self.has_normalization_params = False
        else:
            # Transform vertex delta mean and scale into a tensor.
            self.vertex_delta_mean_cpu[0] = delta_mean.x
            self.vertex_delta_mean_cpu[1] = delta_mean.y
            self.vertex_delta_mean_cpu[2] = delta_mean.z
            self.vertex_delta_scale_cpu[0] = delta_scale.x
            self.vertex_delta_scale_cpu[1] = delta_scale.y
            self.vertex_delta_scale_cpu[2] = delta_scale.z
            print('Statistics summary:')
            print('Vertex delta mean: {}'.format(self.vertex_delta_mean_cpu))
            print('Vertex delta scale: {}'.format(self.vertex_delta_scale_cpu))
    
        return True

    def _internal_getitem_in_params(self, device, pin_memory):
        input_params = []

        # Concatenate the bone rotations.
        if len(self.ml_deformer_dataset.sample_bone_rotations) > 0:
            input_params.extend(self.ml_deformer_dataset.sample_bone_rotations)

        # Concatenate curve values to network's input parameters.
        if len(self.ml_deformer_dataset.sample_curve_values) > 0:
            input_params.extend(self.ml_deformer_dataset.sample_curve_values)

        # Stop training if input parameters are empty.
        if len(input_params) == 0:
            raise StopIteration()

        # Get local params transformation.
        if self.has_normalization_params:
            # Generate random noise if any on the fly.
            self.params = get_params_1d(self.opt.preprocess, len(input_params),
                                        mean=0.0, std=1.0, device=device)

        params_transform = get_transform_1d(self.opt.preprocess, params=self.params,
                                            noise_factor=self.opt.noise_factor,
                                            precision=self.opt.precision, device=device)

        # Apply transformations to parameters and points to get tensor arrays.
        in_params = params_transform(input_params)
        if device.type == 'cpu' and pin_memory and torch.cuda.is_available():
            try:
                in_params.pin_memory()
            except:
                in_params_mem = in_params.nelement() * in_params.element_size()
                print("in_params mem {0}  type {1} device {2}".format(in_params_mem, type(in_params),
                                                                      in_params.is_cuda))
                raise
            
        return in_params

    def _internal_getitem_outvec(self, device, pin_memory):
        """ Critical and potentially slow operation to get delta vector used for training 
        """
        vertex_delta_mean = self.vertex_delta_mean_cpu.to(device, copy=True)
        vertex_delta_scale = self.vertex_delta_scale_cpu.to(device, copy=True)

        points_params = get_params_1d(
            'none', 0, mean=vertex_delta_mean, std=vertex_delta_scale, device=device)
        device_points_transform = get_transform_1d(
            'none', params=points_params, make_points=True,
            precision=self.precision, device=device)

        out_vec = device_points_transform(self.ml_deformer_dataset.sample_deltas)
        if device.type == 'cpu' and pin_memory and torch.cuda.is_available():
            try:
                out_vec.pin_memory()
            except:
                out_vec_mem = out_vec.nelement() * out_vec.element_size()
                print("out_vec_mem {0}  type {1} device {2}".format(out_vec_mem, type(out_vec),
                                                                    out_vec.is_cuda))
                raise
        return out_vec 

    def _internal_getitem(self, index, device, pin_memory):
        """ Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains input and output
            in_vec (tensor)  -- input vector
            out_vec (tensor) -- output (gt) vector
        """
        with torch.autograd.profiler.record_function("UEDefDataset.__getitem__.set_index"):
            sample_exists = self.ml_deformer_dataset.set_current_sample_index(index)

            if not sample_exists:
                raise StopIteration()

            in_params = self._internal_getitem_in_params(device, pin_memory)
            out_vec = self._internal_getitem_outvec(device, pin_memory)
            
            return {'in_params': in_params, 'out_vec': out_vec, 'path': self.opt.dataroot}

        def __len__(self):
            """ Return the total number of samples in the dataset. """
            return self.ml_deformer_dataset.num_samples()


class DeviceCachedDataset(BaseUEDeformerDataset):
    """ Custom dataset that caches all data on GPU or CPU - can only be used if enough memory on device """

    def __init__(self, ml_deformer_dataset, opt):
        """ Initialize this dataset class. """
        BaseUEDeformerDataset.__init__(self,  ml_deformer_dataset, opt)

        # Get device name: CPU or CUDA
        if opt.gpu_ids:
            self.device = torch.device('cuda:{}'.format(opt.gpu_ids[0]))
        else:
            self.device = torch.device('cpu')

        self.cached_data = dict()
        self.use_cache = opt.memory_cache_in_megabytes != 0
        self.total_memory = 0
        self.max_memory = opt.memory_cache_in_megabytes * 1024 * 1024

    def __getitem__(self, index):
        if not self.use_cache:
            return self._internal_getitem(index, self.device, False)
        else:
            if index in self.cached_data:
                return self.cached_data[index]
            else:
                if self.total_memory < self.max_memory:
                    item = self._internal_getitem(index)
                    out_vec_mem = item['out_vec'].nelement() * item['out_vec'].element_size()
                    in_params_mem = item['in_params'].nelement() * item['in_params'].element_size()
                    self.total_memory += out_vec_mem + in_params_mem
                    self.cached_data[index] = item
                    return item
                else:
                    return self._internal_getitem(index, self.device, False)


class CPUPinCachedDataset(BaseUEDeformerDataset):
    """ Data set that caches memory pinned memory on the cpu, and transfers to the device. """

    def __init__(self, ml_deformer_dataset, opt):
        """ Initialize this dataset class. """
        BaseUEDeformerDataset.__init__(self, ml_deformer_dataset, opt)

        # Get device name: CPU or CUDA
        if opt.gpu_ids:
            self.device = torch.device('cuda:{}'.format(opt.gpu_ids[0]))
        else:
            self.device = torch.device('cpu')

        self.cpu_device = torch.device('cpu')
        self.cached_data = dict()
        # noise modifies the inputs each epoch, so it is not possible to cache
        self.use_cache = opt.memory_cache_in_megabytes != 0 and self.opt.preprocess != 'add_noise'
        self.total_memory = 0
        self.max_memory = opt.memory_cache_in_megabytes * 1024 * 1024

    def transfer_cpu_to_device(self, item):
        if self.device.type != 'cpu':
            return {'in_params': item['in_params'].to(non_blocking=True, device=self.device, copy=True),
                    'out_vec':  item['out_vec'].to(non_blocking=True, device=self.device, copy=True),
                    'path': item['path']}
        else:
            return item
        
    def __getitem__(self, index):
        if not self.use_cache:
            return self._internal_getitem(index, self.device, False)
        else:
            if index in self.cached_data:
                return self.transfer_cpu_to_device(self.cached_data[index])
            else: 
                if self.total_memory < self.max_memory:
                    # Always get the item on the CPU and pin it when caching
                    item = self._internal_getitem(index, self.cpu_device, True)
                    out_vec_mem = item['out_vec'].nelement() * item['out_vec'].element_size()
                    in_params_mem = item['in_params'].nelement() * item['in_params'].element_size()
                    self.total_memory += out_vec_mem + in_params_mem 
                    self.cached_data[index] = item
                    return self.transfer_cpu_to_device(item)
                else:
                    # if we can't use the cache, go ahead and load directly on the device
                    return self._internal_getitem(index, self.device, False)

    def __len__(self):
        """ Return the total number of samples in the dataset. """
        return self.ml_deformer_dataset.num_samples()

