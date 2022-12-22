from ModuleConverterInterface import ModuleConverterInterface
import unreal as ue
from unreal import FXConverterUtilitiesLibrary as ueFxUtils


class CascadeMeshMaterialConverter(ModuleConverterInterface):

    @classmethod
    def get_input_cascade_module(cls):
        return ue.ParticleModuleMeshMaterial

    @classmethod
    def convert(cls, args):
        cascade_module = args.get_cascade_module()
        emitter = args.get_niagara_emitter_context()

        # mesh material is only valid for the mesh renderer, so start by finding that.
        mesh_renderer_props = emitter.find_renderer("MeshRenderer")
        if mesh_renderer_props is not None:
            # get all properties from the cascade mesh material module.
            # noinspection PyTypeChecker
            mesh_materials = ueFxUtils.get_particle_module_mesh_material_props(cascade_module)

            # set the mesh overide materials from the casade module properties.
            # noinspection PyTypeChecker
            ueFxUtils.set_mesh_renderer_material_overrides_from_cascade(mesh_renderer_props,  mesh_materials)
