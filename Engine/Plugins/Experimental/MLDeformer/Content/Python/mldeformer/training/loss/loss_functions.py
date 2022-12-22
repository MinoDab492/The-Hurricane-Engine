# -*- coding: utf-8 -*-
"""
This module contains different loss functions.

Copyright Epic Games, Inc. All Rights Reserved.
"""

import torch
import torch.nn as nn


class ShrinkageLoss(nn.Module):
    """Shrinkage loss.

        Modified version of shrinkage loss tailored to vertex data:
        http://openaccess.thecvf.com/content_ECCV_2018/papers/Xiankai_Lu_Deep_Regression_Tracking_ECCV_2018_paper.pdf
        It basically computes a vertex-wise shrinkage loss.
    """

    def __init__(self, shrink_speed: float = 10.0, shrink_loc: float = 0.1):
        """ Initialize the ShrinkageLoss class.

        Parameters:
            shrink_speed_ (float) -- shrinkage speed, i.e., weight assigned to hard samples.
            shrink_loc_ (float)   -- shrinkage localization, i.e., threshold for hard mining.
        """
        super(ShrinkageLoss, self).__init__()
        self.shrink_speed = shrink_speed
        self.shrink_loc = shrink_loc
        print('ShrinkageLoss parameters:')
        print('Speed(slope): {} -- Localization(bias): {}'.format(self.shrink_speed, self.shrink_loc))

    def forward(self, prediction: torch.Tensor, ground_truth: torch.Tensor):
        """ Calculate shrinkage loss between estimated and ground truth deltas.

        Parameters:
            prediction (tensor)   -- predicted deltas.
            ground_truth (tensor) -- ground truth deltas (values should ideally be bounded, for instance, 
                                       in [0,1] or [-1,1] so that meaningful hyperparameters can be selected).

        Returns:
            mean per-vertex shrinkage loss.
        """
        total_size = prediction.nelement()
        
        # Compute pixel errors (l2 norm).
        l2_loss = torch.norm(
            (prediction - ground_truth).view(total_size // 3, 3),
            p=2, dim=1)
        
        # Compute mean shrinkage loss.
        shrink_loss = torch.mul(l2_loss, l2_loss) / (1.0 + torch.exp(self.shrink_speed * (self.shrink_loc - l2_loss)))
        return torch.mean(shrink_loss)


class L2pLoss(nn.Module):
    """ L2pLoss class.
        Extended pixel loss introduced by Thies et al.:
        http://www.graphics.stanford.edu/~niessner/papers/2016/1facetoface/thies2016face.pdf
        This loss computes a vertex-wise L2 norm. Then, it computes the element-wise p 
        power and finally averages out the result.
    """

    def __init__(self, p: float = 1.0):
        """ Initialize L2pLoss with user-defined parameters.

        Parameters:
            p_ (float) -- Target exponent in [0.5,2].
        """
        super(L2pLoss, self).__init__()
        self.p = p
        assert(self.p >= 0.5 and self.p <= 2)

    def forward(self, prediction: torch.Tensor, ground_truth: torch.Tensor):
        """ Calculate L_2^p loss between estimated and ground truth deltas.

        Parameters:
            prediction (tensor)  -- predicted deltas.
            ground_truth (tensor) -- ground truth deltas  (values should ideally be bounded, for instance, 
                                       in [0,1] or [-1,1] so that meaningful hyperparameters can be selected).

        Returns:
            mean per-vertex l2^p loss (float)
        """
        total_size = prediction.nelement()
        
        # Compute L2 norm.
        l2_norm = torch.norm(
            (prediction - ground_truth).view(total_size // 3, 3),
            p=2, dim=1)
        
        # Compute p power and average result.
        if self.p == 1:
            return torch.mean(l2_norm)
        else:
            return torch.mean(torch.pow(l2_norm, self.p))
