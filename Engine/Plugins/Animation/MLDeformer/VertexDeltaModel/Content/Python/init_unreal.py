# -*- coding: utf-8 -*-
"""
VertexDeltaTrainingModel class.

Copyright Epic Games, Inc. All Rights Reserved.
"""

from importlib import reload
import unreal

import vertexdeltamodel

@unreal.uclass()
class VertexDeltaPythonTrainingModel(unreal.VertexDeltaTrainingModel):
    @unreal.ufunction(override=True)
    def train(self):
        reload(vertexdeltamodel)
        return vertexdeltamodel.train(self)
        