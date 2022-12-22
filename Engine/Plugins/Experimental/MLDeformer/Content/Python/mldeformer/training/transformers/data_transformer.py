# -*- coding: utf-8 -*-
"""
This module includes common transformation functions (e.g., get_transform*, 
unnormalize_points, __add_noise*, __to_tensor) that are used by dataset classes.

Copyright Epic Games, Inc. All Rights Reserved.
"""

from __future__ import print_function

import torch
import torchvision.transforms as transforms

from mldeformer.utils.misc.data_converter import list2tensor


def unnormalize_points(in_array, points_center, points_std, unflatten_array=False):
    """Un-normalize data by adding center and divide by scale.

    Parameters:
        in_array (ndarray) -- 1D or 2D array.
        points_center (ndarray) -- Points center.
        points_std (ndarray/scalar) -- (An)Isotropic std dev.
        unflatten_array (bool) -- Whether the array must be unflattened before normalizing.
    Return:
        Output array (ndarray)
    """
    if unflatten_array:
        # Unflat and un-normalize array.
        out_array = in_array.view(in_array.nelement() // 3, 3) * points_std + points_center
        
        # Flat array again.
        return out_array.flatten()   
    else:
        # Un-normalize array.
        return in_array * points_std + points_center


# Vector transformation
def get_params_1d(preprocess, size, std_devs=torch.empty(0), mean=0.0, std=1.0, device='cpu'):
    noise = torch.empty(0)
    if preprocess == 'add_noise' and size > 0:
        noise = 2.0 * torch.rand(size).to(device) - 1.0
        if std_devs.nelement() == 1 or std_devs.nelement() == noise.nelement():
            noise = torch.mul(std_devs, noise)

    return {'noise': noise, 'mean': mean, 'std': std}


def get_transform_1d(
    preprocess, params=None, noise_factor=0.01, convert=True, make_points=False, precision='float', device='cpu'):
    transform_list = []

    if convert:
        if params is not None:
            transform_list += [transforms.Lambda(
                lambda vec: __to_tensor(vec, make_points, precision, device))]
            transform_list += [transforms.Lambda(
                lambda vec: __normalize_1d(vec, params['mean'], params['std'], True))]
        else:
            transform_list += [transforms.Lambda(
                lambda vec: __to_tensor(vec, False, precision, device))]
    else:
        transform_list += [transforms.Lambda(
            lambda vec: __to_tensor(vec, False, precision, device))]

    if 'add_noise' in preprocess:
        if params is None:
            transform_list.append(transforms.Lambda(lambda vec: __add_noise_1d(vec, torch.empty(0), noise_factor)))
        else:
            transform_list.append(transforms.Lambda(lambda vec: __add_noise_1d(vec, params['noise'], noise_factor)))

    return transforms.Compose(transform_list)


def __to_tensor(vec, make_points, precision, device):
    if make_points:
        tensor_vec = list2tensor(vec, precision).to(device)
        return tensor_vec.view(tensor_vec.nelement() // 3, 3)
    else:
        return list2tensor(vec, precision).to(device)


def __normalize_1d(vec, mean, std, flatten):
    with torch.autograd.profiler.record_function("data_transformer.__normalize_1d"):
        if type(std) == torch.Tensor:
            assert torch.norm(std) > 1e-5
        else:
            assert std > 1e-5
        if flatten:
            return ((vec - mean) / std).flatten()
        else:
            return (vec - mean) / std


def __add_noise_1d(vec, noise, factor):
    if noise.nelement() == 1 or noise.nelement() == vec.nelement():
        return vec + factor * noise
    else:
        return vec + factor * torch.mul(
            vec, 2.0 * torch.rand(vec.nelement()).to(vec.device) - 1.0)
