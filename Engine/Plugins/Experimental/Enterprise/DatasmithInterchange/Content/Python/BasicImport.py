import os
import unreal

editor_actor_subsytem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)

actor = editor_actor_subsytem.spawn_actor_from_class(unreal.Actor, unreal.Vector())

import_context = unreal.DatasmithInterchangeImportContext()
import_context.asset_path = "/Game/Python/Datasmith/Interchange/Assets/"
import_context.anchor = actor.get_editor_property('root_component')
import_context.async_ = False
import_context.static_mesh_options.remove_degenerates = False

tessel_options = unreal.DatasmithCommonTessellationOptions()
tessel_options.options.chord_tolerance = 20.0

import_context.import_options.append(tessel_options)

result = unreal.DatasmithInterchangeScripting.load_file("D:/Models/SLDWKS/mouse_01.SLDPRT", import_context)

