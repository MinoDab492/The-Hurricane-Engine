# -*- coding: utf-8 -*-
"""
This module contains simple io functions to convert mesh files.

Copyright Epic Games, Inc. All Rights Reserved.
"""

from __future__ import print_function

import os


def mkdirs(paths):
    """Create empty directories if they don't exist.

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """Create a single empty directory if it didn't exist.

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)


def rmdir(path):
    """Remove folder and its files, if there exist any.

    Parameters:
        path (str) -- a single directory path
    """
    if os.path.exists(path) and os.path.isdir(path):
        # List all files in directory
        for f in os.listdir(path):
            fpath = os.path.join(path, f)
            if os.path.isfile(fpath):
                os.remove(fpath)
        os.rmdir(path)


def rmfiles(path):
    """Remove all files in a folder, if there exist any.

    Parameters:
        path (str) -- a single directory path
    """
    if os.path.exists(path) and os.path.isdir(path):
        # List all files in directory
        for f in os.listdir(path):
            fpath = os.path.join(path, f)
            if os.path.isfile(fpath):
                os.remove(fpath)
