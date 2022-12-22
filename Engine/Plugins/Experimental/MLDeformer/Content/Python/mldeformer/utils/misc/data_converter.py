# -*- coding: utf-8 -*-
"""
This module contains simple functions to convert data structures.

Copyright Epic Games, Inc. All Rights Reserved.
"""

from __future__ import print_function

import numpy as np
import torch


def list2tensor(array, precision='float'):
    """"Converts a list into a tensor array.

    Parameters:
        array (list) -- input numpy array
        precision (str)    -- precision type [double | float | half]
    Return:
        tensor array
    """
    with torch.autograd.profiler.record_function("utils.misc.data_converter.list2tensor"):
        # Convert input numpy array into a tensor.
        if precision == 'float':
            return torch.tensor(array).float()
        elif precision == 'double':
            return torch.tensor(array).double()
        elif precision == 'half':
            return torch.tensor(array).half()
        else:
            return torch.tensor(array).float()


def numpy2tensor(array, precision='float'):
    """"Converts a numpy array into a tensor array.

    Parameters:
        array (np.ndarray) -- input numpy array
        precision (str)    -- precision type [double | float | half]
    Return:
        tensor array
    """

    with torch.autograd.profiler.record_function("utils.misc.data_converter.numpy2tensor"):
        # Convert input numpy array into a tensor.
        if precision == 'float':
            return torch.from_numpy(array).float()
        elif precision == 'double':
            return torch.from_numpy(array).double()
        elif precision == 'half':
            return torch.from_numpy(array).half()
        else:
            return torch.from_numpy(array).float()


def tensor2numpy(array, precision='float'):
    """"Converts a Tensor array into a numpy array.

    Parameters:
        array (Tensor)  -- input tensor array
        precision (str) -- precision type [double | float | half]
    Return:
        numpy array
    """
    if not isinstance(array, np.ndarray):
        if isinstance(array, torch.Tensor):
            # Convert input tensor into a numpy array.
            if precision == 'float':
                return array.data[0].cpu().float().numpy()
            elif precision == 'double':
                return array.data[0].cpu().double().numpy()
            elif precision == 'half':
                return array.data[0].cpu().half().numpy()
            else:
                return array.data[0].cpu().float().numpy()
        else:
            return array
    else:  # if it is a numpy array, do nothing
        return array
