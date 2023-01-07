from ModuleConverterInterface import ModuleConverterInterface
import unreal as ue
from unreal import FXConverterUtilitiesLibrary as ueFxUtils
import CascadeToNiagaraHelperMethods as c2nUtils
import Paths


class CascadeColorScaleConverter(ModuleConverterInterface):

    @classmethod
    def get_input_cascade_module(cls):
        return ue.ParticleModuleColorScaleOverLife

    @classmethod
    def convert(cls, args):
        cascade_module = args.get_cascade_module()
        emitter = args.get_niagara_emitter_context()

        # make the new module script for color scale over life
        script_asset = ueFxUtils.create_asset_data(Paths.script_color_scale)
        script_args = ue.CreateScriptContextArgs(script_asset)
        color_scale_script = emitter.find_or_add_module_script(
            "ColorScaleOverLife",
            script_args,
            ue.ScriptExecutionCategory.PARTICLE_UPDATE)

        # get all properties from the cascade color scale over life module
        # noinspection PyTypeChecker
        (color_scale_over_life_distribution,
         alpha_scale_over_life_distribution,
         b_emitter_time
         ) = ueFxUtils.get_particle_module_color_scale_over_life_props(cascade_module)

        # convert the color scale and alpha scale over life properties
        options = c2nUtils.DistributionConversionOptions()

        # if sampling curves with emitter age time, replace the curve index
        if b_emitter_time is True:
            emitter_time_input = ueFxUtils.create_script_input_linked_parameter(
                "Emitter.NormalizedLoopAge",
                ue.NiagaraScriptInputType.FLOAT)
            options.set_custom_indexer(emitter_time_input)

        color_scale_over_life_input = c2nUtils.create_script_input_for_distribution(
            color_scale_over_life_distribution,
            options)

        alpha_scale_over_life_input = c2nUtils.create_script_input_for_distribution(
            alpha_scale_over_life_distribution,
            options)

        # set the color and alpha scale
        color_scale_script.set_parameter("Scale RGB", color_scale_over_life_input)
        color_scale_script.set_parameter("Scale Alpha", alpha_scale_over_life_input)
