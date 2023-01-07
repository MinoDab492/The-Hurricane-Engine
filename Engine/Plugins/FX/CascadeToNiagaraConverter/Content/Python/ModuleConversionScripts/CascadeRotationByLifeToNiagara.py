from ModuleConverterInterface import ModuleConverterInterface
import unreal as ue
from unreal import FXConverterUtilitiesLibrary as ueFxUtils
import CascadeToNiagaraHelperMethods as c2nUtils
import Paths


class CascadeRotationByLifeConverter(ModuleConverterInterface):

    @classmethod
    def get_input_cascade_module(cls):
        return ue.ParticleModuleRotationOverLifetime

    @classmethod
    #  todo handle mesh rotation rate
    def convert(cls, args):
        cascade_module = args.get_cascade_module()
        emitter = args.get_niagara_emitter_context()
        
        # get all properties from the cascade rotation over life module
        # noinspection PyTypeChecker
        (rotation_over_life_distribution,
         b_scale
         ) = ueFxUtils.get_particle_module_rotation_over_lifetime_props(cascade_module)

        # make an input to apply the rotation
        rotation_input = c2nUtils.create_script_input_for_distribution(
            rotation_over_life_distribution)

        # choose how to apply the rotation.
        # from ParticleModuleRotationOverLifetime.h:
        # If true, the particle rotation is multiplied by the value retrieved from RotationOverLife.
        # If false, the particle rotation is incremented by the value retrieved from RotationOverLife.
        
        cur_rotation_input = ueFxUtils.create_script_input_linked_parameter(
            "Particles.SpriteRotation",
            ue.NiagaraScriptInputType.FLOAT)
        if b_scale:
            script_asset = ueFxUtils.create_asset_data(Paths.di_multiply_float)
            script_args = ue.CreateScriptContextArgs(script_asset)
            combine_script = ueFxUtils.create_script_context(script_args)
        else:
            script_asset = ueFxUtils.create_asset_data(Paths.di_add_floats)
            script_args = ue.CreateScriptContextArgs(script_asset)
            combine_script = ueFxUtils.create_script_context(script_args)
            
        combine_script.set_parameter("A", cur_rotation_input)
        combine_script.set_parameter("B", rotation_input)
        final_rotation_input = ueFxUtils.create_script_input_dynamic(combine_script, ue.NiagaraScriptInputType.FLOAT)
        
        # convert the combined rotation to degrees
        script_asset = ueFxUtils.create_asset_data(Paths.di_angle_conversion)
        script_args = ue.CreateScriptContextArgs(script_asset)
        norm_angle_to_degrees_script = ueFxUtils.create_script_context(script_args)
        norm_angle_to_degrees_script.set_parameter(
            "Angle Input",
            ueFxUtils.create_script_input_enum(Paths.enum_niagara_angle_input, "Normalized Angle (0-1)"))
        norm_angle_to_degrees_script.set_parameter(
            "Angle Output",
            ueFxUtils.create_script_input_enum(Paths.enum_niagara_angle_input, "Degrees"))
        norm_angle_to_degrees_script.set_parameter("Angle", final_rotation_input)
        final_rotation_input = ueFxUtils.create_script_input_dynamic(
            norm_angle_to_degrees_script,
            ue.NiagaraScriptInputType.FLOAT)
        
        # set the sprite rotation parameter directly
        emitter.set_parameter_directly(
            "Particles.SpriteRotation",
            final_rotation_input,
            ue.ScriptExecutionCategory.PARTICLE_UPDATE)

        # find/add the module script for particle state
        script_args = ue.CreateScriptContextArgs(ueFxUtils.create_asset_data(Paths.script_particle_state), [1, 1])
        emitter.find_or_add_module_script("ParticleState", script_args, ue.ScriptExecutionCategory.PARTICLE_UPDATE)
