# -*- coding: utf-8 -*-
"""
Neural Morph Model.

Copyright Epic Games, Inc. All Rights Reserved.
"""

from importlib import reload
import unreal
import neuralmorphmodel_local
import neuralmorphmodel_global

@unreal.uclass()
class NeuralMorphPythonTrainingModel(unreal.NeuralMorphTrainingModel):
    @unreal.ufunction(override=True)
    def train(self):
        if self.get_model().mode == 0:
            reload(neuralmorphmodel_local)
            return neuralmorphmodel_local.train(self)
        else:
            assert self.get_model().mode == 1
            reload(neuralmorphmodel_global)
            return neuralmorphmodel_global.train(self)
