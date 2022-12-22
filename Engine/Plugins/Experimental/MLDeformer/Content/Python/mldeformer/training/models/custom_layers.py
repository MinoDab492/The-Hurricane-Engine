# -*- coding: utf-8 -*-
"""
This module contains custom layers.

Copyright Epic Games, Inc. All Rights Reserved.
"""

import torch.nn as nn


class Identity(nn.Module):
    def forward(self, x):
        return x
