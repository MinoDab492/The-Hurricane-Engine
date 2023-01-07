from ModuleConverterInterface import ModuleConverterInterface
import unreal as ue
from unreal import FXConverterUtilitiesLibrary as ueFxUtils
import CascadeToNiagaraHelperMethods as c2nUtils
import Paths


class CascadeColorOverLifeConverter(ModuleConverterInterface):

    @classmethod
    def get_input_cascade_module(cls):
        return ue.ParticleModuleColorOverLife

    @classmethod
    def convert(cls, args):
        cascade_module = args.get_cascade_module()
        emitter = args.get_niagara_emitter_context()
        
        # make the new module script for color over life
        script_asset = ueFxUtils.create_asset_data(Paths.script_color)
        script_args = ue.CreateScriptContextArgs(script_asset)
        color_script = emitter.find_or_add_module_script(
            "ColorOverLife",
            script_args,
            ue.ScriptExecutionCategory.PARTICLE_UPDATE)

        # get all properties from the cascade color over life module
        # noinspection PyTypeChecker
        (color_over_life_distribution,
         alpha_over_life_distribution,
         b_clamp_alpha
         ) = ueFxUtils.get_particle_module_color_over_life_props(cascade_module)
        
        # convert the color over life property
        color_over_life_input = c2nUtils.create_script_input_for_distribution(color_over_life_distribution)
        
        # convert the alpha over life property
        alpha_over_life_input = c2nUtils.create_script_input_for_distribution(alpha_over_life_distribution)
        
        # convert the clamp alpha property
        if b_clamp_alpha is True:
            script_asset = ueFxUtils.create_asset_data(Paths.di_clamp_float)
            script_args = ue.CreateScriptContextArgs(script_asset)
            clamp_float_script = ueFxUtils.create_script_context(script_args)
            min_input = ueFxUtils.create_script_input_float(0)
            max_input = ueFxUtils.create_script_input_float(1)
            clamp_float_script.set_parameter("Min", min_input)
            clamp_float_script.set_parameter("Max", max_input)
            
            # reassign the alpha over life input to the new clamped alpha over
            # life input so that it is applied to the top level color script
            clamp_float_script.set_parameter("Float", alpha_over_life_input)
            alpha_over_life_input = ueFxUtils.create_script_input_dynamic(
                clamp_float_script,
                ue.NiagaraScriptInputType.FLOAT)
        
        # combine color over life and alpha over life into linear color
        script_asset = ueFxUtils.create_asset_data(Paths.di_color_from_vec_and_float)
        script_args = ue.CreateScriptContextArgs(script_asset)
        break_color_script = ueFxUtils.create_script_context(script_args)
        break_color_script.set_parameter("Vector (RGB)", color_over_life_input)
        break_color_script.set_parameter("Float (Alpha)", alpha_over_life_input)
        
        # set the color
        break_color_script_input = ueFxUtils.create_script_input_dynamic(
            break_color_script,
            ue.NiagaraScriptInputType.LINEAR_COLOR)
        color_script.set_parameter("Color", break_color_script_input)
