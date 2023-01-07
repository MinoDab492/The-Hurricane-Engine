# -*- coding: utf-8 -*-
"""
NearestNeighborTrainingModel class.

Copyright Epic Games, Inc. All Rights Reserved.
"""

from importlib import reload
import unreal

import nearestneighbormodel

@unreal.uclass()
class NearestNeighborPythonTrainingModel(unreal.NearestNeighborTrainingModel):
    @unreal.ufunction(override=True)
    def train(self):
        reload(nearestneighbormodel)
        return nearestneighbormodel.train(self)

    @unreal.ufunction(override=True)
    def update_nearest_neighbor_data(self):
        reload(nearestneighbormodel)
        return nearestneighbormodel.update_nearest_neighbor_data(self)

    @unreal.ufunction(override=True)
    def kmeans_cluster_poses(self):
        reload(nearestneighbormodel)
        return nearestneighbormodel.kmeans_cluster_poses(self)