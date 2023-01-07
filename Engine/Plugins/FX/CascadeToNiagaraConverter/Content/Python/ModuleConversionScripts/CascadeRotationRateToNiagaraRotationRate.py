from ModuleConverterInterface import ModuleConverterInterface
import unreal as ue
from unreal import FXConverterUtilitiesLibrary as ueFxUtils
import CascadeToNiagaraHelperMethods as c2nUtils
import Paths


class CascadeRotationRateConverter(ModuleConverterInterface):

    @classmethod
    def get_input_cascade_module(cls):
        return ue.ParticleModuleRotationRate

    @classmethod
    #  todo handle mesh rotation rate
    def convert(cls, args):
        cascade_module = args.get_cascade_module()
        emitter = args.get_niagara_emitter_context()
        
        # get all properties from the cascade rotation module
        # noinspection PyTypeChecker
        rotation = ueFxUtils.get_particle_module_rotation_rate_props(cascade_module)

        # make an input to apply the rotation rate
        options = c2nUtils.DistributionConversionOptions()
        options.set_target_type_width(ue.NiagaraScriptInputType.FLOAT)
        rotation_input = c2nUtils.create_script_input_for_distribution(rotation, options)

        # set the sprite rotation rate parameter directly
        emitter.set_parameter_directly(
            "Particles.SpriteRotationRate",
            rotation_input,
            ue.ScriptExecutionCategory.PARTICLE_SPAWN)
        
        # find/add a module script for sprite rotation rate
        script_asset = ueFxUtils.create_asset_data(Paths.script_sprite_rotation_rate)
        script_args = ue.CreateScriptContextArgs(script_asset)
        rotation_rate_script = emitter.find_or_add_module_script(
            "SpriteRotationRate",
            script_args,
            ue.ScriptExecutionCategory.PARTICLE_UPDATE)
        
        # get the rotation rate from spawn
        rotation_from_spawn_input = ueFxUtils.create_script_input_linked_parameter(
            "Particles.SpriteRotationRate",
            ue.NiagaraScriptInputType.FLOAT)
        
        # convert the rotation rate to degrees
        script_asset = ueFxUtils.create_asset_data(Paths.di_angle_conversion)
        script_args = ue.CreateScriptContextArgs(script_asset)
        norm_angle_to_degrees_script = ueFxUtils.create_script_context(script_args)
        norm_angle_to_degrees_script.set_parameter(
            "Angle Input",
            ueFxUtils.create_script_input_enum(Paths.enum_niagara_angle_input, "Normalized Angle (0-1)"))
        norm_angle_to_degrees_script.set_parameter(
            "Angle Output",
            ueFxUtils.create_script_input_enum(Paths.enum_niagara_angle_input, "Degrees"))
        norm_angle_to_degrees_script.set_parameter("Angle", rotation_from_spawn_input)
        rotation_from_spawn_input = ueFxUtils.create_script_input_dynamic(
            norm_angle_to_degrees_script,
            ue.NiagaraScriptInputType.FLOAT)
        
        rotation_rate_script.set_parameter("Rotation Rate", rotation_from_spawn_input)

        # find/add the module script for particle state
        script_args = ue.CreateScriptContextArgs(ueFxUtils.create_asset_data(Paths.script_particle_state), [1, 1])
        emitter.find_or_add_module_script("ParticleState", script_args, ue.ScriptExecutionCategory.PARTICLE_UPDATE)