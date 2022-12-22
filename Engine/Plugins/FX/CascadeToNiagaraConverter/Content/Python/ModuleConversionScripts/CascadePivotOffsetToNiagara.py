from ModuleConverterInterface import ModuleConverterInterface
import unreal as ue
from unreal import FXConverterUtilitiesLibrary as ueFxUtils


class CascadePivotOffsetConverter(ModuleConverterInterface):

    @classmethod
    def get_input_cascade_module(cls):
        return ue.ParticleModulePivotOffset

    @classmethod
    def convert(cls, args):
        cascade_module = args.get_cascade_module()
        emitter = args.get_niagara_emitter_context()

        # pivot offset is only valid for the sprite renderer, so start by finding that.
        sprite_renderer_props = emitter.find_renderer("SpriteRenderer")
        if sprite_renderer_props is not None:
            # noinspection PyTypeChecker
            # get the pivot offset module properties.
            pivot_offset = ueFxUtils.get_particle_module_pivot_offset_props(cascade_module)
            pivot_offset = pivot_offset * -1.0
            
            # set the pivot offset on the sprite renderer.
            sprite_renderer_props.set_editor_property("PivotInUVSpace", pivot_offset)
