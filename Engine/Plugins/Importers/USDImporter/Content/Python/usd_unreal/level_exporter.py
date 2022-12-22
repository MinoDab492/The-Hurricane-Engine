from collections import defaultdict
from pxr import Usd, UsdUtils, UsdGeom, Sdf, UsdLux, Gf, UsdSkel, UsdShade, Tf
import os
import re
import sys
import shutil
import unreal
import math
from timeit import default_timer as timer
from pathlib import Path
from usd_unreal.exporting_utils import *
from usd_unreal.generic_utils import *
from usd_unreal.constants import *

static_mesh_editor_subsytem = unreal.get_editor_subsystem(unreal.StaticMeshEditorSubsystem)

def collect_actors(context):
    """ Collects a list of actors to export according to `context`'s options

    :param context: UsdExportContext object describing the export
    :returns: Set of unreal.Actor objects that were collected
    """
    actors = None
    if context.options.inner.selection_only:
        # Go through LayersSubsystem as EditorLevelLibrary has too aggressive filtering
        layers_subsystem = unreal.LayersSubsystem()
        actors = set(layers_subsystem.get_selected_actors())
    else:
        actors = unreal.UsdConversionLibrary.get_actors_to_convert(context.world)
    actors = set([a for a in actors if a])

    # Each sublevel has an individual (mostly unused) world, that actually has the name of the sublevel
    # on the UE editor. The levels themselves are all named "Persistent Level", so we can't filter based on them
    if len(context.options.inner.levels_to_ignore) > 0:
        persistent_level_allowed = "Persistent Level" not in context.options.inner.levels_to_ignore

        filtered_actors = set()
        for actor in actors:
            # If the actor is in a sublevel, this will return the persistent world instead
            persistent_world = actor.get_world()

            # If the actor is in a sublevel, this will be the (mostly unused) world that actually owns the sublevel
            actual_world = actor.get_outer().get_outer()

            actor_in_persistent_level = actual_world == persistent_world
            sublevel_name = actual_world.get_name()

            # Have to make sure we only allow it via sublevel name if the actor is not on the persistent level,
            # because if it is then the level name will be the name of the ULevel asset (and not actually "Persistent Level")
            # and we'd incorrectly let it through
            if (persistent_level_allowed and actor_in_persistent_level) or (not actor_in_persistent_level and sublevel_name not in context.options.inner.levels_to_ignore):
                filtered_actors.add(actor)

        actors = set(filtered_actors)

    actors = [a for a in actors if should_export_actor(a)]
    return actors


def collect_materials(actors, static_meshes, skeletal_meshes):
    """ Collects all unreal.MaterialInterface objects that are used by meshes and as overrides in unreal.MeshComponents

    :param actors: Iterable of unreal.Actor objects to traverse looking for unreal.MeshComponents material overrides
    :param static_meshes: Iterable of unreal.StaticMesh objects to collect used materials from
    :param skeletal_meshes: Iterable of unreal.SkeletalMesh objects to collect used materials from
    :returns: Set of unreal.MaterialInterface objects that were collected
    """
    materials = set()

    # Collect materials used as component overrides
    visited_components = set()

    # Don't capture materials or visited_components here as this function definition may live
    # indefinitely and prevent garbage collection
    def traverse_components(component, materials, visited_components):
        if not component or component in visited_components:
            return
        visited_components.add(component)

        actor = component.get_owner()
        if not should_export_actor(actor):
            return

        if isinstance(component, unreal.MeshComponent):
            for mat in component.get_editor_property('override_materials'):
                if mat:
                    materials.add(mat)

        if isinstance(component, unreal.SceneComponent):
            for child in component.get_children_components(include_all_descendants=False):
                traverse_components(child, materials, visited_components)

    for actor in actors:
        traverse_components(actor.get_editor_property("root_component"), materials, visited_components)

    # Collect materials used by static meshes
    for mesh in static_meshes:
        for static_material in mesh.get_editor_property('static_materials'):
            mat = static_material.get_editor_property('material_interface')
            if mat:
                materials.add(mat)

    # Collect materials used by skeletal meshes
    for mesh in skeletal_meshes:
        for skeletal_material in mesh.get_editor_property('materials'):
            mat = skeletal_material.get_editor_property('material_interface')
            if mat:
                materials.add(mat)

    return materials


def collect_static_meshes(actors):
    """ Collects all static meshes that are used by unreal.StaticMeshComponents of actors

    :param actors: Iterable of unreal.Actor
    :returns: Set of unreal.StaticMesh objects
    """
    meshes = set()

    for actor in actors:
        for comp in actor.get_components_by_class(unreal.StaticMeshComponent.static_class()):
            # Really we want to check bIsVisualizationComponent, so that we can skip exporting sprites
            # and camera meshes and directional light arrows and so on, but that's not exposed to blueprint
            # so this is the best we can do
            if comp.is_editor_only:
                continue

            mesh = comp.get_editor_property('static_mesh')
            if mesh:
                meshes.add(mesh)

        if isinstance(actor, unreal.InstancedFoliageActor):
            for foliage_type in actor.get_used_foliage_types():
                source = foliage_type.get_source()
                if isinstance(source, unreal.StaticMesh):
                    meshes.add(source)

    return meshes


def collect_skeletal_meshes(actors):
    """ Collects all skeletal meshes that are used by unreal.SkinnedMeshComponents of actors

    :param actors: Iterable of unreal.Actor
    :returns: Set of unreal.SkeletalMesh objects
    """
    meshes = set()

    for actor in actors:
        for comp in actor.get_components_by_class(unreal.SkinnedMeshComponent.static_class()):
            mesh = comp.get_editor_property('skeletal_mesh')
            if mesh:
                meshes.add(mesh)

    return meshes


def collect_cubemap_textures(actors):
    """ Collects all unreal.TextureCube assets that are used by unreal.SkyLightComponents

    :param actors: Iterable of unreal.Actor
    :returns: Set of unreal.TextureCube assets
    """
    cubemaps = set()

    for actor in actors:
        for comp in actor.get_components_by_class(unreal.SkyLightComponent.static_class()):
            # Really we want to check bIsVisualizationComponent, so that we can skip exporting sprites
            # and camera meshes and directional light arrows and so on, but that's not exposed to blueprint
            # so this is the best we can do
            if comp.is_editor_only:
                continue

            cubemap = comp.get_editor_property('cubemap')
            if cubemap:
                cubemaps.add(cubemap)

    return cubemaps


def assign_static_mesh_component_assets(component, prim, exported_assets):
    """ Assigns the reference, exported static mesh asset for this static mesh component

    :param component: unreal.StaticMeshComponent to export
    :param prim: Usd.Prim that will reference the Mesh prim asset usd file
    :param exported_assets: Maps from unreal.Object to exported filename (e.g. "C:/MyFolder/Mesh.usda")
                            Used to retrieve the exported filename of the unreal.StaticMesh used by the component, if any
    :returns: None
    """
    mesh = component.get_editor_property('static_mesh')
    if not mesh:
        return

    asset_file_path = exported_assets[mesh]
    if asset_file_path is None:
        unreal.log_warning(f"Failed to find an exported usd file to use for mesh asset '{mesh.get_name()}'")
        return

    add_relative_reference(prim, asset_file_path)


def assign_hism_component_assets(component, prim, exported_assets, material_overrides_layer):
    """ Assigns the reference, exported static mesh asset for this hierarchical instanced static mesh component

    :param component: unreal.HierarchicalInstancedStaticMeshComponent to export
    :param prim: Usd.Prim that contains the UsdGeom.PointInstancer schema
    :param exported_assets: Maps from unreal.Object to exported filename (e.g. "C:/MyFolder/Mesh.usda")
                            Used to retrieve the exported filename of the unreal.StaticMesh used by the component, if any
    :param material_overrides_layer: Sdf.Layer to author the material overrides in
    :returns: None
    """
    point_instancer = UsdGeom.PointInstancer(prim)
    if not point_instancer:
        return

    mesh = component.get_editor_property('static_mesh')
    if not mesh:
        return

    asset_file_path = exported_assets[mesh]
    if asset_file_path is None:
        unreal.log_warning(f"Failed to find an exported usd file to use for mesh asset '{mesh.get_name()}'")
        return

    stage = prim.GetStage()
    if not stage:
        return

    # Create prototypes Scope prim
    prototypes = stage.DefinePrim(prim.GetPath().AppendPath('Prototypes'), 'Scope')

    # Create child proxy prims to reference the exported meshes
    # Add proxy prims to prototypes relationship in order
    rel = point_instancer.CreatePrototypesRel()

    exported_filepath = exported_assets[mesh]
    filename = Path(exported_filepath).stem
    proxy_prim_name = Tf.MakeValidIdentifier(filename)
    proxy_prim_name = get_unique_name(set(), proxy_prim_name)

    # Don't use 'Mesh' if we have LODs because it makes it difficult to parse back the prototypes if
    # they're nested Meshes
    proxy_prim_schema = 'Mesh' if static_mesh_editor_subsytem.get_lod_count(mesh) < 2 else ''

    proxy_prim = stage.DefinePrim(prototypes.GetPath().AppendPath(proxy_prim_name), proxy_prim_schema)
    add_relative_reference(proxy_prim, exported_filepath)

    rel.AddTarget(proxy_prim.GetPath())

    # Set material overrides if we have any
    mat_overrides = component.get_editor_property('override_materials')
    with Usd.EditContext(stage, material_overrides_layer):
        apply_static_mesh_material_overrides(mat_overrides, proxy_prim, mesh)


def assign_skinned_mesh_component_assets(component, prim, exported_assets):
    """ Assigns the reference, exported skeletal mesh asset for this skeletal mesh component

    :param component: unreal.SkinnedMeshComponent to export
    :param prim: Usd.Prim that contains the UsdSkel.Root schema
    :param exported_assets: Maps from unreal.Object to exported filename (e.g. "C:/MyFolder/Mesh.usda")
                            Used to retrieve the exported filename of the unreal.SkeletalMesh used by the component, if any
    :returns: None
    """
    skel_root = UsdSkel.Root(prim)
    if not skel_root:
        return

    mesh = component.get_editor_property('skeletal_mesh')
    if not mesh:
        return

    asset_file_path = exported_assets[mesh]
    if asset_file_path is None:
        unreal.log_warning(f"Failed to find an exported usd file to use for mesh asset '{mesh.get_name()}'")
        return

    add_relative_reference(prim, asset_file_path)


def assign_instanced_foliage_actor_assets(actor, prim, exported_assets, material_overrides_layer):
    """ Assigns the reference, exported static mesh assets for this unreal.InstancedFoliageActor

    :param actor: unreal.InstancedFoliageActor to convert
    :param prim: Usd.Prim with the UsdGeom.PointInstancer schema
    :param exported_assets: Maps from unreal.Object to exported filename (e.g. "C:/MyFolder/Mesh.usda")
                            Used to retrieve the exported filename of the unreal.StaticMeshes used by actor, if any
    :param material_overrides_layer: Sdf.Layer to author the material overrides in
    :returns: None
    """
    point_instancer = UsdGeom.PointInstancer(prim)
    if not point_instancer:
        return

    if not isinstance(actor, unreal.InstancedFoliageActor):
        return

    stage = prim.GetStage()
    if not stage:
        return

    # Get a packed, sorted list of foliage types that correspond to meshes we exported
    types = actor.get_used_foliage_types()
    exported_types = []
    exported_type_sources = []
    for foliage_type in types:
        source = foliage_type.get_source()
        if isinstance(source, unreal.StaticMesh) and source in exported_assets:
            exported_types.append(foliage_type)
            exported_type_sources.append(source)

    # Create prototypes Scope prim
    prototypes = stage.DefinePrim(prim.GetPath().AppendPath('Prototypes'), 'Scope')

    # Create child proxy prims to reference the exported meshes
    # Add proxy prims to prototypes relationship in order
    rel = point_instancer.CreatePrototypesRel()
    unique_child_prims = set()
    for foliage_type, source in zip(exported_types, exported_type_sources):
        exported_filepath = exported_assets[source]

        filename = Path(exported_filepath).stem
        proxy_prim_name = Tf.MakeValidIdentifier(filename)
        proxy_prim_name = get_unique_name(unique_child_prims, proxy_prim_name)
        unique_child_prims.add(proxy_prim_name)

        # Don't use 'Mesh' if we have LODs because it makes it difficult to parse back the prototypes if
        # they're nested Meshes
        proxy_prim_schema = 'Mesh' if static_mesh_editor_subsytem.get_lod_count(source) < 2 else ''

        proxy_prim = stage.DefinePrim(prototypes.GetPath().AppendPath(proxy_prim_name), proxy_prim_schema)
        add_relative_reference(proxy_prim, exported_filepath)

        rel.AddTarget(proxy_prim.GetPath())

        # Set material overrides if we have any
        if isinstance(foliage_type, unreal.FoliageType_InstancedStaticMesh):
            mat_overrides = foliage_type.get_editor_property('override_materials')
            with Usd.EditContext(stage, material_overrides_layer):
                apply_static_mesh_material_overrides(mat_overrides, proxy_prim, source)


def assign_material_prim(prim, material_prim):
    """ Assigns material_prim as a material binding to prim, traversing LODs instead, if available

    :param prim: Usd.Prim to receive the material binding. It optionally contains a LOD variant set
    :param material_prim: Usd.Prim with the UsdShade.Material schema that will be bound as a material
    :returns: None
    """
    stage = prim.GetStage()

    var_set = prim.GetVariantSet(LOD_VARIANT_SET_TOKEN)
    lods = var_set.GetVariantNames()
    if len(lods) > 0:
        original_selection = var_set.GetVariantSelection() if var_set.HasAuthoredVariantSelection() else None

        # Switch into each of the LOD variants the prim has, and recurse into the child prims
        for variant in lods:
            with Usd.EditContext(stage, stage.GetSessionLayer()):
                var_set.SetVariantSelection(variant)

            with var_set.GetVariantEditContext():
                for child in prim.GetChildren():
                    if UsdGeom.Mesh(child):
                        mat_binding_api = UsdShade.MaterialBindingAPI.Apply(child)
                        mat_binding_api.Bind(UsdShade.Material(material_prim))

        # Restore the variant selection to what it originally was
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            if original_selection:
                var_set.SetVariantSelection(original_selection)
            else:
                var_set.ClearVariantSelection()
    else:
        mat_binding_api = UsdShade.MaterialBindingAPI.Apply(prim)
        mat_binding_api.Bind(UsdShade.Material(material_prim))


def convert_component(context, converter, component, visited_components=set(), parent_prim_path=""):
    """ Exports a component onto the stage, creating the required prim of the corresponding schema as necessary

    :param context: UsdExportContext object describing the export
    :param component: unreal.SceneComponent to export
    :param visited_components: Set of components to skip during traversal. Exported components are added to it
    :param parent_prim_path: Prim path the parent component was exported to (e.g. "/Root/Parent").
                             Purely an optimization: Can be left empty, and will be discovered automatically
    :returns: None
    """
    actor = component.get_owner()
    if not should_export_actor(actor):
        return

    # We use this as a proxy for bIsVisualizationComponent
    if component.is_editor_only:
        return

    prim_path = unreal.UsdConversionLibrary.get_prim_path_for_object(
        component,
        parent_prim_path,
        context.options.inner.export_actor_folders
    )

    prim_path = get_unique_name(context.exported_prim_paths, prim_path)
    context.exported_prim_paths.add(prim_path)

    unreal.log(f"Exporting component '{component.get_name()}' onto prim '{prim_path}'")

    material_overrides_layer = context.materials_sublayer if context.materials_sublayer else context.stage.GetRootLayer()

    prim = context.stage.GetPrimAtPath(prim_path)
    if not prim:
        target_schema_name = unreal.UsdConversionLibrary.get_schema_name_for_component(component)
        unreal.log(f"Creating new prim at '{prim_path}' with schema '{target_schema_name}'")
        prim = context.stage.DefinePrim(prim_path, target_schema_name)

    # Add attributes to prim depending on component type
    if isinstance(component, unreal.HierarchicalInstancedStaticMeshComponent):
        # Create a separate new child of `prim` that can receive our PointInstancer component schema instead.
        # We do this because we can have any component tree in UE, but in USD the recommendation is that you don't
        # place drawable prims inside PointInstancers, and that DCCs don't traverse PointInstancers looking for drawable
        # prims, so that they can work as a way to "hide" their prototypes
        child_prim_path = get_unique_name(context.exported_prim_paths, prim_path + "/HISMInstance")
        context.exported_prim_paths.add(child_prim_path)

        unreal.log(f"Creating new prim at '{child_prim_path}' with schema 'PointInstancer'")
        child_prim = context.stage.DefinePrim(child_prim_path, "PointInstancer")

        assign_hism_component_assets(component, child_prim, context.exported_assets, material_overrides_layer)
        converter.convert_hism_component(component, child_prim_path)

    elif isinstance(component, unreal.StaticMeshComponent):
        assign_static_mesh_component_assets(component, prim, context.exported_assets)

        # Author material overrides on the material overrides layer:
        old_edit_target = converter.get_edit_target()
        try:
            converter.set_edit_target(unreal.FilePath(material_overrides_layer.realPath))
            converter.convert_mesh_component(component, prim_path)
        finally:
            converter.set_edit_target(old_edit_target)

    elif isinstance(component, unreal.SkinnedMeshComponent):
        assign_skinned_mesh_component_assets(component, prim, context.exported_assets)

        # Author material overrides on the material overrides layer:
        old_edit_target = converter.get_edit_target()
        try:
            converter.set_edit_target(unreal.FilePath(material_overrides_layer.realPath))
            converter.convert_mesh_component(component, prim_path)
        finally:
            converter.set_edit_target(old_edit_target)

    elif isinstance(component, unreal.CineCameraComponent):

        # If we're the main camera component of an ACineCameraActor, then write that out on our parent prim
        # so that if we ever import this file back into UE we can try to reconstruct the ACineCameraActor
        # with the same root and camera components, instead of creating new ones
        owner_actor = component.get_owner()
        if isinstance(owner_actor, unreal.CineCameraActor):
            main_camera_component = owner_actor.get_editor_property("camera_component")
            if component == main_camera_component:
                parent_prim = prim.GetParent()
                if parent_prim:
                    attr = parent_prim.CreateAttribute('unrealCameraPrimName', Sdf.ValueTypeNames.Token)
                    attr.Set(prim.GetName())

        converter.convert_cine_camera_component(component, prim_path)

    elif isinstance(component, unreal.LightComponentBase):
        converter.convert_light_component(component, prim_path)

        if isinstance(component, unreal.SkyLightComponent):
            converter.convert_sky_light_component(component, prim_path)

        if isinstance(component, unreal.DirectionalLightComponent):
            converter.convert_directional_light_component(component, prim_path)

        if isinstance(component, unreal.RectLightComponent):
            converter.convert_rect_light_component(component, prim_path)

        if isinstance(component, unreal.PointLightComponent):
            converter.convert_point_light_component(component, prim_path)

            if isinstance(component, unreal.SpotLightComponent):
                converter.convert_spot_light_component(component, prim_path)

    if isinstance(component, unreal.SceneComponent):
        converter.convert_scene_component(component, prim_path)

        owner_actor = component.get_owner()

        # We have to export the instanced foliage actor in one go, because it will contain one component
        # for each foliage type, and we don't want to end up with one PointInstancer prim for each
        if isinstance(owner_actor, unreal.InstancedFoliageActor):
            assign_instanced_foliage_actor_assets(actor, prim, context.exported_assets, material_overrides_layer)
            level = None if context.options.inner.export_foliage_on_actors_layer else actor.get_level()
            converter.convert_instanced_foliage_actor(actor, prim_path, level)

        elif isinstance(owner_actor, unreal.LandscapeProxy):
            success, mesh_path = export_landscape(context, owner_actor)
            if success:
                add_relative_reference(prim, mesh_path)
            else:
                unreal.log_warning(f"Failed to export landscape '{owner_actor.get_name()}' to filepath '{mesh_path}'")
        else:
            # Recurse to children
            for child in component.get_children_components(include_all_descendants=False):
                if child in visited_components:
                    continue
                visited_components.add(child)

                convert_component(context, converter, child, visited_components, prim_path)


def export_actors(context, actors):
    """ Will export the `actors`' component hierarchies as prims on the context's stage

    :param context: UsdExportContext object describing the export
    :param actors: Collection of unreal.Actor objects to iterate over and export
    :returns: None
    """
    unreal.log(f"Exporting components from {len(actors)} actors")

    visited_components = set()

    with UsdConversionContext(context.root_layer_path) as converter:

        # Component traversal ensures we parse parent components before child components,
        # but since we get our actors in random order we need to manually ensure we parse
        # parent actors before child actors. Otherwise we may parse a child, end up having USD create
        # all the parent prims with default Xform schemas to make the child path work, and then
        # not being able to convert a parent prim to a Mesh schema (or some other) because a
        # default prim already exists with that name, and it's a different schema.
        def attach_depth(a):
            depth = 0
            parent = a.get_attach_parent_actor()
            while parent:
                depth += 1
                parent = parent.get_attach_parent_actor()
            return depth
        actor_list = list(actors)
        actor_list.sort(key=attach_depth)

        for actor in actor_list:
            comp = actor.get_editor_property("root_component")
            if not comp or comp in visited_components:
                continue
            visited_components.add(comp)

            # If this actor is in a sublevel, make sure the prims are authored in the matching sub-layer
            with ScopedObjectEditTarget(actor, context):
                this_edit_target = context.stage.GetEditTarget().GetLayer().realPath
                converter.set_edit_target(unreal.FilePath(this_edit_target))

                convert_component(context, converter, comp, visited_components)


def setup_material_override_layer(context):
    """ Creates a dedicated layer for material overrides, and adds it as a sublayer to all layers in the layer stack.

    We use a dedicated layer to author material overrides because if a prim is defined in a stage with multiple layers,
    USD will only actually `def` the prim on the weakest layer, and author `over`s on stronger layers. We don't want
    this behavior for our material prims, because then we wouldn't be able to open each layer as an independent stage.
    We may want to be able to do that, as UE levels are fully independent from eachother.

    By keeping all material overrides in a separate layer that all other layers have as sublayer, we can open each one
    independently, or the whole stage as a group, and the material assignments are correct.

    :param context: UsdExportContext object describing the export
    :returns: Sdf.Layer that should contain the material override prims
    """

    root_layer_folder = os.path.dirname(context.root_layer_path)
    mat_layer_path = os.path.join(root_layer_folder, "MaterialOverrides" + context.scene_file_extension)

    # Create as a stage so that it owns the reference to its root layer, and is predictably closed when it runs out of
    # scope. If we open the layer directly a strong reference to it will linger and prevent it from being collected
    mat_layer_stage = Usd.Stage.CreateNew(mat_layer_path)

    for layer in reversed(context.stage.GetLayerStack(includeSessionLayers=False)):
        unreal.UsdConversionLibrary.insert_sub_layer(layer.realPath, mat_layer_path)

    return mat_layer_stage.GetRootLayer()


def export_level(context, actors):
    """ Exports the actors and components of the level to the main output root layer, and potentially sublayers

    This function creates the root layer for the export, and then iterates the component attachment hierarchy of the
    current level, creating a prim for each exportable component.

    :param context: UsdExportContext object describing the export
    :returns: None
    """
    unreal.log(f"Creating new stage with root layer '{context.root_layer_path}'")

    with unreal.ScopedSlowTask(3, f"Exporting main root layer") as slow_task:
        slow_task.make_dialog(True)

        slow_task.enter_progress_frame(1)

        # Setup stage
        context.stage = Usd.Stage.CreateNew(context.root_layer_path)
        root_prim = context.stage.DefinePrim('/' + ROOT_PRIM_NAME, 'Xform')
        context.stage.SetDefaultPrim(root_prim)
        root_layer = context.stage.GetRootLayer()

        # Set stage metadata
        UsdGeom.SetStageUpAxis(context.stage, UsdGeom.Tokens.z if context.options.stage_options.up_axis == unreal.UsdUpAxis.Z_AXIS else UsdGeom.Tokens.y)
        UsdGeom.SetStageMetersPerUnit(context.stage, context.options.stage_options.meters_per_unit)
        if context.options.start_time_code != 0.0:
            context.stage.SetStartTimeCode(context.options.start_time_code)
        if context.options.end_time_code != 0.0:
            context.stage.SetEndTimeCode(context.options.end_time_code)

        # Prepare sublayers for export, if applicable
        if context.options.inner.export_sublayers:
            levels = set([a.get_level() for a in actors])
            context.level_to_sublayer = create_a_sublayer_for_each_level(context, levels)
            context.materials_sublayer = setup_material_override_layer(context)

        # Export actors
        export_actors(context, actors)

        # Post-processing
        slow_task.enter_progress_frame(1)
        if context.options.inner.asset_options.bake_materials:
            override_layer = context.materials_sublayer if context.materials_sublayer else root_layer

            # Replace material overrides with baked versions, if any
            with UsdConversionContext(context.root_layer_path) as converter:
                converter.replace_unreal_materials_with_baked(
                    unreal.FilePath(override_layer.realPath),
                    context.baked_materials,
                    is_asset_layer=False,
                    use_payload=context.options.inner.asset_options.use_payload,
                    remove_unreal_materials=context.options.inner.asset_options.remove_unreal_materials
                )

            # Abandon the material overrides sublayer if we don't have any material overrides
            if context.materials_sublayer and context.materials_sublayer.empty:
                # Cache this path because the materials sublayer may get suddenly dropped as we erase all sublayer references
                materials_sublayer_path = context.materials_sublayer.realPath
                sublayers = context.stage.GetLayerStack(includeSessionLayers=False)
                sublayers.remove(context.materials_sublayer)

                for sublayer in sublayers:
                    relative_path = unreal.UsdConversionLibrary.make_path_relative_to_layer(sublayer.realPath, materials_sublayer_path)
                    sublayer.subLayerPaths.remove(relative_path)
                os.remove(materials_sublayer_path)

        # Write file
        slow_task.enter_progress_frame(1)
        unreal.log(f"Saving root layer '{context.root_layer_path}'")
        if context.options.inner.export_sublayers:
            # Use the layer stack directly so we also get a material sublayer if we made one
            for sublayer in context.stage.GetLayerStack(includeSessionLayers=False):
                sublayer.Save()
        context.stage.GetRootLayer().Save()


def export_material(context, material):
    """ Exports a single unreal.MaterialInterface

    :param context: UsdExportContext object describing the export
    :param material: unreal.MaterialInterface object to export
    :returns: (Bool, String) containing True if the export was successful, and the output filename that was used
    """
    material_file = get_filename_to_export_to(context.options.inner.asset_folder.path, material, context.scene_file_extension)

    # If we try to bake a material instance without a valid parent, the material baking module will hard crash the editor
    if isinstance(material, unreal.MaterialInstance):
        parent = material.get_editor_property('parent')
        if not parent:
            unreal.log_warning(f"Failed to export material '{material.get_name()}' to filepath '{material_file}': Material instance has no parent!")
            return (False, material_file)

    options = unreal.MaterialExporterUSDOptions()
    options.material_baking_options = context.options.inner.asset_options.material_baking_options
    options.material_baking_options.textures_dir = unreal.DirectoryPath(os.path.join(os.path.dirname(material_file), 'Textures'))

    task = unreal.AssetExportTask()
    task.object = material
    task.filename = material_file
    task.selected = False
    task.replace_identical = True
    task.prompt = False
    task.options = options
    task.automated = True

    unreal.log(f"Exporting material '{material.get_name()}' to filepath '{material_file}'")
    success = unreal.Exporter.run_asset_export_task(task)

    if not success:
        unreal.log_warning(f"Failed to export material '{material.get_name()}' to filepath '{material_file}'")
        return (False, material_file)

    try:
        # Add USD info to exported asset
        stage = Usd.Stage.Open(material_file)
        usd_prim = stage.GetDefaultPrim()
        model = Usd.ModelAPI(usd_prim)
        model.SetAssetIdentifier(material_file)
        model.SetAssetName(material.get_name())

        # We should remove any 'unreal' render context surface output from the baked materials if
        # remove_unreal_materials was checked, otherwise we'll still get the unreal materials on the level
        # if opening this scene back in the editor, as the material asset files also now reference the UE assets
        # by default too
        if context.options.inner.asset_options.remove_unreal_materials and usd_prim:
            with UsdConversionContext(material_file) as converter:
                converter.remove_unreal_surface_output(
                    str(usd_prim.GetPrimPath()),
                    unreal.FilePath(material_file)
                )

        stage.Save()
    except BaseException as e:
        if len(material_file) > 220:
            unreal.log_error(f"USD failed to open a stage with a very long filepath: Try to use a destination folder with a shorter file path")
        unreal.log_error(e)

    return (True, material_file)


def export_materials(context, materials):
    """ Exports a collection of unreal.MaterialInteface objects

    :param context: UsdExportContext object describing the export
    :param materials: Collection of unreal.MaterialInterface objects to export
    :returns: None
    """
    if not context.options.inner.asset_options.bake_materials:
        return

    num_materials = len(materials)
    unreal.log(f"Baking {len(materials)} materials")

    with unreal.ScopedSlowTask(num_materials, 'Baking materials') as slow_task:
        slow_task.make_dialog(True)

        for material in materials:
            if slow_task.should_cancel():
                break
            slow_task.enter_progress_frame(1)

            success, filename = export_material(context, material)
            if success:
                context.exported_assets[material] = filename
                context.baked_materials[material.get_path_name()] = filename


def export_mesh(context, mesh, mesh_type):
    """ Exports a single Static/Skeletal mesh

    :param context: UsdExportContext object describing the export
    :param mesh: unreal.SkeletalMesh or unreal.StaticMesh object to export
    :param mesh_type: String 'static' or 'skeletal', according to the types of `meshes`
    :returns: (Bool, String) containing True if the export was successful, and the output filename that was used
    """
    mesh_file = get_filename_to_export_to(context.options.inner.asset_folder.path, mesh, context.scene_file_extension)

    options = None
    if mesh_type == 'static':
        options = unreal.StaticMeshExporterUSDOptions()
    elif mesh_type == 'skeletal':
        options = unreal.SkeletalMeshExporterUSDOptions()

    options.stage_options = context.options.stage_options
    options.mesh_asset_options = context.options.inner.asset_options

    # Force this to false because we don't want the static mesh exporter to bake our materials,
    # as we will do that ourselves with varying TextureDirs
    options.mesh_asset_options.bake_materials = False

    task = unreal.AssetExportTask()
    task.object = mesh
    task.filename = mesh_file
    task.selected = False
    task.replace_identical = True
    task.prompt = False
    task.automated = True
    task.options = options

    unreal.log(f"Exporting {mesh_type} mesh '{mesh.get_name()}' to filepath '{mesh_file}'")
    success = unreal.Exporter.run_asset_export_task(task)

    if not success:
        unreal.log_warning(f"Failed to export {mesh_type} mesh '{mesh.get_name()}' to filepath '{mesh_file}'")
        return (False, mesh_file)

    # Add USD info to exported asset
    try:
        stage = Usd.Stage.Open(mesh_file)
        usd_prim = stage.GetDefaultPrim()
        model = Usd.ModelAPI(usd_prim)
        model.SetAssetIdentifier(mesh_file)
        model.SetAssetName(mesh.get_name())

        # Take over material baking for the mesh, so that it's easier to share
        # baked materials between mesh exports
        if context.options.inner.asset_options.bake_materials:
            with UsdConversionContext(mesh_file) as converter:
                converter.replace_unreal_materials_with_baked(
                    unreal.FilePath(stage.GetRootLayer().realPath),
                    context.baked_materials,
                    is_asset_layer=True,
                    use_payload=context.options.inner.asset_options.use_payload,
                    remove_unreal_materials=context.options.inner.asset_options.remove_unreal_materials
                )

        stage.Save()
    except BaseException as e:
        if len(mesh_file) > 220:
            unreal.log_error(f"USD failed to open a stage with a very long filepath: Try to use a destination folder with a shorter file path")
        unreal.log_error(e)

    return (True, mesh_file)


def export_meshes(context, meshes, mesh_type):
    """ Exports a collection of Static/Skeletal meshes

    :param context: UsdExportContext object describing the export
    :param meshes: Homogeneous collection of unreal.SkeletalMesh or unreal.StaticMesh
    :param mesh_type: String 'static' or 'skeletal', according to the types of `meshes`
    :returns: None
    """
    num_meshes = len(meshes)
    unreal.log(f"Exporting {num_meshes} {mesh_type} meshes")

    with unreal.ScopedSlowTask(num_meshes, f"Exporting {mesh_type} meshes") as slow_task:
        slow_task.make_dialog(True)

        for mesh in meshes:
            if slow_task.should_cancel():
                break
            slow_task.enter_progress_frame(1)

            success, path = export_mesh(context, mesh, mesh_type)
            if success:
                context.exported_assets[mesh] = path


def export_cubemap(context, cubemap):
    """ Exports an unreal.TextureCube as an hdr map

    :param context: UsdExportContext object describing the export
    :param material: unreal.TextureCube object to export
    :returns: (Bool, String) containing True if the export was successful, and the output filename that was used
    """
    if not cubemap:
        return

    hdr_file = get_filename_to_export_to(context.options.inner.asset_folder.path, cubemap, ".hdr")

    task = unreal.AssetExportTask()
    task.object = cubemap
    task.filename = hdr_file
    task.selected = False
    task.replace_identical = True
    task.prompt = False
    task.automated = True

    unreal.log(f"Exporting cubemap '{cubemap.get_name()}' to filepath '{hdr_file}'")
    success = unreal.Exporter.run_asset_export_task(task)

    if not success:
        unreal.log_warning(f"Failed to export cubemap '{cubemap.get_name()}' to filepath '{hdr_file}'")
        return (False, hdr_file)

    return (True, hdr_file)


def export_cubemaps(context, cubemaps):
    """ Exports a collection of unreal.TextureCube assets

    :param context: UsdExportContext object describing the export
    :param materials: Collection of unreal.TextureCube objects to export
    :returns: None
    """
    num_cubemaps = len(cubemaps)
    unreal.log(f"Copying/exporting {num_cubemaps} TextureCubes")

    with unreal.ScopedSlowTask(num_cubemaps, 'Exporting/copying cubemaps') as slow_task:
        slow_task.make_dialog(True)

        for cubemap in cubemaps:
            if slow_task.should_cancel():
                break
            slow_task.enter_progress_frame(1)

            asset_import_data = cubemap.get_editor_property('asset_import_data')
            if not asset_import_data:
                asset_import_data = unreal.AssetImportData(cubemap)
                cubemap.set_editor_property('asset_import_data', asset_import_data)

            # Just copy over the cubemap to the output folder
            existing_filename = asset_import_data.get_first_filename()
            if os.path.exists(existing_filename) and os.path.isfile(existing_filename):
                target_filename = get_filename_to_export_to(context.options.inner.asset_folder.path, cubemap, ".hdr")

                # It's already at our desired destination, so we must have already handled it for another component
                if os.path.exists(target_filename) and os.path.samefile(target_filename, existing_filename):
                    continue

                unreal.log(f"Copying existing hdr texture '{existing_filename}' to filepath '{target_filename}'")

                # shutil.copy2 will fail if the folder doesn't exist so let's make sure it does
                target_dir = os.path.dirname(target_filename)
                if not os.path.exists(target_dir):
                    os.makedirs(target_dir)

                shutil.copy2(existing_filename, target_filename)

                context.cubemap_original_filenames[cubemap] = existing_filename
                asset_import_data.scripted_add_filename(target_filename, 0, "")

            # Source hdr map is not available: Export TextureCube to a new hdr
            else:
                success, exported_filename = export_cubemap(context, cubemap)
                if success:
                    # Keep track of the original asset import data filename, but stash our exported
                    # filename there so that converter.convert_sky_light_component will check it and use it
                    # when binding the HDR texture onto the DomeLight
                    context.cubemap_original_filenames[cubemap] = asset_import_data.get_first_filename()
                    asset_import_data.scripted_add_filename(exported_filename, 0, "")


def export_landscape(context, landscape_actor):
    """ Exports a landscape actor as separate mesh and material USD files referencing eachother, and returns the path to the mesh file

    :param context: UsdExportContext object describing the export
    :param landscape_actor: unreal.LandscapeProxy actor to export
    :returns: (Bool, String) containing True if the export was successful, and the output mesh filename that was used
    """
    folder_path = get_actor_asset_folder_to_export_to(context.options.inner.asset_folder.path, landscape_actor)
    actor_name = sanitize_filepath(landscape_actor.get_name())

    textures_folder = os.path.join(folder_path, "Textures")
    asset_path = os.path.join(folder_path, actor_name + "_asset" + context.scene_file_extension)
    mesh_path = os.path.join(folder_path, actor_name + ("_payload." + context.options.inner.asset_options.payload_format if context.options.inner.asset_options.use_payload else "_mesh" + context.scene_file_extension))
    mat_path = os.path.join(folder_path, actor_name + "_material" + context.scene_file_extension)

    # Use the UsdUtils stage cache because we'll use the C++ wrapper function to add a payload,
    # and we want to fetch the same stage
    cache = UsdUtils.StageCache.Get()
    with Usd.StageCacheContext(cache):
        # Create stage for the material
        mat_stage = Usd.Stage.CreateNew(mat_path)
        mat_prim = mat_stage.DefinePrim('/' + Tf.MakeValidIdentifier(actor_name), 'Material')
        mat_stage.SetDefaultPrim(mat_prim)

        # Set material stage metadata
        UsdGeom.SetStageUpAxis(mat_stage, UsdGeom.Tokens.z if context.options.stage_options.up_axis == unreal.UsdUpAxis.Z_AXIS else UsdGeom.Tokens.y)
        UsdGeom.SetStageMetersPerUnit(mat_stage, context.options.stage_options.meters_per_unit)
        mat_stage.SetStartTimeCode(context.options.start_time_code)
        mat_stage.SetEndTimeCode(context.options.end_time_code)
        model = Usd.ModelAPI(mat_prim)
        model.SetAssetIdentifier(mat_path)
        model.SetAssetName(actor_name)

        # Convert material data
        properties_to_bake = [
            unreal.PropertyEntry(property_=unreal.MaterialProperty.MP_BASE_COLOR),
            unreal.PropertyEntry(property_=unreal.MaterialProperty.MP_METALLIC),
            unreal.PropertyEntry(property_=unreal.MaterialProperty.MP_SPECULAR),
            unreal.PropertyEntry(property_=unreal.MaterialProperty.MP_ROUGHNESS),
            unreal.PropertyEntry(property_=unreal.MaterialProperty.MP_NORMAL)
        ]
        with UsdConversionContext(mat_path) as converter:
            converter.convert_landscape_proxy_actor_material(
                landscape_actor,
                str(mat_prim.GetPath()),
                properties_to_bake,
                context.options.inner.landscape_bake_resolution,
                unreal.DirectoryPath(textures_folder)
            )

        # Create stage for the mesh
        mesh_stage = Usd.Stage.CreateNew(mesh_path)
        num_lods = max(abs(context.options.inner.highest_landscape_lod - context.options.inner.lowest_landscape_lod + 1), 1)
        mesh_prim = mesh_stage.DefinePrim('/' + Tf.MakeValidIdentifier(actor_name), 'Mesh' if num_lods == 1 else "")
        mesh_stage.SetDefaultPrim(mesh_prim)

        # Set mesh stage metadata
        UsdGeom.SetStageUpAxis(mesh_stage, UsdGeom.Tokens.z if context.options.stage_options.up_axis == unreal.UsdUpAxis.Z_AXIS else UsdGeom.Tokens.y)
        UsdGeom.SetStageMetersPerUnit(mesh_stage, context.options.stage_options.meters_per_unit)
        mesh_stage.SetStartTimeCode(context.options.start_time_code)
        mesh_stage.SetEndTimeCode(context.options.end_time_code)
        model = Usd.ModelAPI(mesh_prim)
        model.SetAssetIdentifier(mesh_path)
        model.SetAssetName(actor_name)

        # Convert mesh data
        with UsdConversionContext(mesh_path) as converter:
            converter.convert_landscape_proxy_actor_mesh(
                landscape_actor,
                str(mesh_prim.GetPath()),
                context.options.inner.lowest_landscape_lod,
                context.options.inner.highest_landscape_lod
            )

        # Create a separate "asset" file for the landscape
        asset_stage = None
        if context.options.inner.asset_options.use_payload:
            # Create stage for the mesh
            asset_stage = Usd.Stage.CreateNew(asset_path)
            num_lods = max(abs(context.options.inner.highest_landscape_lod - context.options.inner.lowest_landscape_lod + 1), 1)
            asset_prim = asset_stage.DefinePrim('/' + Tf.MakeValidIdentifier(actor_name), 'Mesh' if num_lods == 1 else "")
            asset_stage.SetDefaultPrim(asset_prim)

            # Set assset stage metadata
            UsdGeom.SetStageUpAxis(asset_stage, UsdGeom.Tokens.z if context.options.stage_options.up_axis == unreal.UsdUpAxis.Z_AXIS else UsdGeom.Tokens.y)
            UsdGeom.SetStageMetersPerUnit(asset_stage, context.options.stage_options.meters_per_unit)
            asset_stage.SetStartTimeCode(context.options.start_time_code)
            asset_stage.SetEndTimeCode(context.options.end_time_code)
            model = Usd.ModelAPI(asset_prim)
            model.SetAssetIdentifier(asset_path)
            model.SetAssetName(actor_name)

            # Refer to the mesh prim as a payload
            unreal.UsdConversionLibrary.add_payload(asset_path, str(asset_prim.GetPath()), mesh_path)

            # Create a material proxy prim and scope on the asset stage
            mat_scope_prim = asset_stage.DefinePrim(mesh_prim.GetPath().AppendChild('Materials'), 'Scope')
            mat_proxy_prim = asset_stage.DefinePrim(mat_scope_prim.GetPath().AppendChild('Material'), 'Material')
            add_relative_reference(mat_proxy_prim, mat_path)

            # Assign material proxy prim to any and all LODs/mesh on the asset stage
            assign_material_prim(asset_prim, mat_proxy_prim)

        # Just export mesh and material files
        else:
            # Create a material proxy prim and scope on the mesh stage
            mat_scope_prim = mesh_stage.DefinePrim(mesh_prim.GetPath().AppendChild('Materials'), 'Scope')
            mat_proxy_prim = mesh_stage.DefinePrim(mat_scope_prim.GetPath().AppendChild('Material'), 'Material')
            add_relative_reference(mat_proxy_prim, mat_path)

            # Assign material proxy prim to any and all LODs/mesh on the mesh stage
            assign_material_prim(mesh_prim, mat_proxy_prim)

        # Write data to files
        mesh_stage.Save()
        mat_stage.Save()
        if asset_stage:
            asset_stage.Save()

        # Remove these stages from the cache or else they will sit there forever
        cache.Erase(mesh_stage)
        cache.Erase(mat_stage)
        if asset_stage:
            cache.Erase(asset_stage)

    return (True, asset_path if context.options.inner.asset_options.use_payload else mesh_path)


def revert_cubemap_import_filenames(cubemap_original_filenames):
    """ Goes over cubemap_original_filenames and reverts each TextureCube's AssetImportData to point to the original
    filename.

    :param cubemap_original_filenames: Maps from unreal.TextureCube to a path that was the original first filename
    on its AssetImportData before we swapped it with our exported file
    :returns: None
    """
    for cubemap, filename in cubemap_original_filenames.items():
        if not cubemap:
            continue

        asset_import_data = cubemap.get_editor_property('asset_import_data')
        if asset_import_data:
            asset_import_data.scripted_add_filename(filename, 0, "")


def export(context):
    """ Exports the current level according to the received export context

    :param context: UsdExportContext object describing the export
    :returns: None
    """
    context.start_time = timer()
    try:
        if not context.world or not isinstance(context.world, unreal.World):
            unreal.log_error("UsdExportContext's 'world' member must point to a valid unreal.World object!")
            return

        # Keep track of the original state of the world's streaming levels
        # We do this because we will force load/show the levels we'll export, to guarantee they're loaded
        # properly and things like material baking work correctly
        loaded_before = unreal.UsdConversionLibrary.get_loaded_level_names(context.world)
        visible_before = unreal.UsdConversionLibrary.get_visible_in_editor_level_names(context.world)

        # Stream in all sublevels of the given world before we proceed anything
        unreal.UsdConversionLibrary.stream_in_required_levels(context.world, context.options.inner.levels_to_ignore)

        if context.options.inner.ignore_sequencer_animations:
            unreal.UsdConversionLibrary.revert_sequencer_animations()

        context.root_layer_path = sanitize_filepath(context.root_layer_path)

        # Provide a default assets folder if we haven't been given one
        if not context.options.inner.asset_folder.path:
            context.options.inner.asset_folder.path = os.path.join(os.path.dirname(context.root_layer_path), "Assets")

        if not context.scene_file_extension:
            extension = os.path.splitext(context.root_layer_path)[1]
            context.scene_file_extension = extension if extension else ".usda"
        if not context.options.inner.asset_options.payload_format:
            context.options.inner.asset_options.payload_format = context.scene_file_extension

        unreal.log(f"Starting export to root layer: '{context.root_layer_path}'")

        with unreal.ScopedSlowTask(6, f"Exporting level to '{context.root_layer_path}'") as slow_task:
            slow_task.make_dialog(True)

            # Collect items to export
            slow_task.enter_progress_frame(1)
            actors = collect_actors(context)
            static_meshes = collect_static_meshes(actors)
            skeletal_meshes = collect_skeletal_meshes(actors)
            materials = collect_materials(actors, static_meshes, skeletal_meshes)
            cubemaps = collect_cubemap_textures(actors)

            # Export assets
            slow_task.enter_progress_frame(1)
            export_materials(context, materials)

            slow_task.enter_progress_frame(1)
            export_meshes(context, static_meshes, 'static')

            slow_task.enter_progress_frame(1)
            export_meshes(context, skeletal_meshes, 'skeletal')

            slow_task.enter_progress_frame(1)
            export_cubemaps(context, cubemaps)

            # Export actors
            slow_task.enter_progress_frame(1)
            export_level(context, actors)

        # Revert world's streaming levels to their original states
        loaded_after = unreal.UsdConversionLibrary.get_loaded_level_names(context.world)
        visible_after = unreal.UsdConversionLibrary.get_visible_in_editor_level_names(context.world)
        levels_to_unload = list(set(loaded_after) - set(loaded_before))
        levels_to_hide = list(set(visible_after) - set(visible_before))
        if levels_to_unload or levels_to_hide:
            unreal.UsdConversionLibrary.stream_out_levels(
                context.world,
                levels_to_unload,
                levels_to_hide
            )

    finally:
        # This prevents a leaked stage reference in case export is called in some type of
        # context where Python doesn't want to garbage collect so soon, and yet we expect
        # to not have stage references anymore (e.g. if we did multiple exports on the same scope
        # using the same context)
        context.stage = None
        context.world = None
        context.level_to_sublayer = {}
        context.materials_sublayer = None

        # Make sure we revert any TextureCube assets modified back to its original AssetImportData state
        revert_cubemap_import_filenames(context.cubemap_original_filenames)

        if context.options.inner.ignore_sequencer_animations:
            unreal.UsdConversionLibrary.reapply_sequencer_animations()

        # Send analytics
        send_analytics(context, actors, static_meshes, skeletal_meshes, materials)

def export_with_cdo_options():
    options_cdo = unreal.get_default_object(unreal.LevelExporterUSDOptions)

    context = UsdExportContext()
    context.root_layer_path = options_cdo.current_task.filename
    context.world = options_cdo.current_task.object
    context.options = options_cdo
    context.automated = options_cdo.current_task.automated

    export(context)
