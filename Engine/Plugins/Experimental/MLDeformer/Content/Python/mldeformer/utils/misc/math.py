# -*- coding: utf-8 -*-
"""
This module contains simple math functions.

Copyright Epic Games, Inc. All Rights Reserved.
"""

from __future__ import print_function

import numpy as np


def degs2rads(degs):
    """"Converts degrees into radians.

    Parameters:
        degs (float/double) -- Degrees
    Return:
        Radians
    """
    return degs * np.pi / 180.0


def rads2degs(rads):
    """"Converts radians into degrees.

    Parameters:
        rads (float/double) -- Radians
    Return:
        Degrees
    """
    return rads * 180.0 / np.pi


def clamp_angle(angle, bound=180.0):
    """"Clamps angle in [-bound,bound].

    Parameters:
        angle (float/double) -- Degrees
        bound (float/double) -- Symmetric boundary, either 180 or 360
    Return:
        Clamped degrees
    """
    if bound == 180.0:
        bound2 = bound * 2
    else:
        bound2 = bound

    while angle < -bound:
        angle = bound2 + angle
        
    while angle > bound:
        angle = angle - bound2

    return angle
