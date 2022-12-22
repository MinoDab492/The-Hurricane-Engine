# -*- coding: utf-8 -*-
"""
Package 'datasets' includes all the modules related to data loading

 To add a custom dataset class called 'dummy', you need to add a file called 'dummy_dataset.py' and define a subclass 'DummyDataset' inherited from BaseDataset.
 You need to implement four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point from data loader.

Now you can use the dataset class by specifying flag '--dataset_mode dummy'.

Copyright Epic Games, Inc. All Rights Reserved.
"""

from __future__ import print_function

import importlib
import torch.utils.data as data
from .unreal_deformer_dataset import CPUPinCachedDataset
from .base_dataset import BaseDataset


def create_dataset(data_interface, opt):
    """Create a dataset given the option.

    This function wraps the class CustomDatasetDataLoader.
        This is the main interface between this package and 'train.py'/'test.py'

    Example:
        from data import create_dataset
        dataset = create_dataset(opt)
    """
    data_loader = CustomDatasetDataLoader(data_interface, opt)
    dataset = data_loader.load_data()
    return dataset


class CustomDatasetDataLoader:
    """Wrapper class of Dataset class that performs multi-threaded data loading"""

    def __init__(self, data_interface, opt):
        """Initialize this class

        Step 1: create a dataset instance given the name [dataset_mode]
        Step 2: create a multi-threaded data loader.
        """
        self.opt = opt
        self.dataset = CPUPinCachedDataset(data_interface, opt)
        print('dataset [{}] was created'.format(type(self.dataset).__name__))
        self.dataloader = data.DataLoader(self.dataset, batch_size=opt.batch_size,
                                          shuffle=not opt.serial_batches,
                                          num_workers=int(opt.num_threads))

    def load_data(self):
        return self

    def __len__(self):
        """Return the number of elements in the dataset"""
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data
