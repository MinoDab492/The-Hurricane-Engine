from ModuleConverterInterface import ModuleConverterInterface
import unreal as ue
from unreal import FXConverterUtilitiesLibrary as ueFxUtils
import CascadeToNiagaraHelperMethods as c2nUtils
import Paths


class CascadeRotationConverter(ModuleConverterInterface):

    @classmethod
    def get_input_cascade_module(cls):
        return ue.ParticleModuleRotation

    @classmethod
    def convert(cls, args):
        cascade_module = args.get_cascade_module()
        emitter = args.get_niagara_emitter_context()
        
        # find/add the module script for init particle
        script_asset = ueFxUtils.create_asset_data(Paths.script_initialize_particle)
        script_args = ue.CreateScriptContextArgs(script_asset, [1, 0])
        initialize_particle_script = emitter.find_or_add_module_script(
            "InitializeParticle",
            script_args,
            ue.ScriptExecutionCategory.PARTICLE_SPAWN)
        
        # get all properties from the cascade rotation module
        # noinspection PyTypeChecker
        rotation = ueFxUtils.get_particle_module_rotation_props(cascade_module)

        # make an input to apply the rotation
        rotation_input = c2nUtils.create_script_input_for_distribution(rotation)

        # set the rotation value
        rotation_mode_input = ueFxUtils.create_script_input_enum(
            Paths.enum_niagara_sprite_rotation_mode,
            "Direct Normalized Angle (0-1)")
        initialize_particle_script.set_parameter("Sprite Rotation Mode", rotation_mode_input)
        initialize_particle_script.set_parameter("Sprite Rotation Angle", rotation_input)
