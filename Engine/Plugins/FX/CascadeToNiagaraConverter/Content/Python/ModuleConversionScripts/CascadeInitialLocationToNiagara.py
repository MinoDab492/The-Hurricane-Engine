from ModuleConverterInterface import ModuleConverterInterface
import unreal as ue
from unreal import FXConverterUtilitiesLibrary as ueFxUtils
import CascadeToNiagaraHelperMethods as c2nUtils
import Paths


class CascadeInitialLocationConverter(ModuleConverterInterface):

    @classmethod
    def get_input_cascade_module(cls):
        return ue.ParticleModuleLocation

    @classmethod
    def convert(cls, args):
        cascade_module = args.get_cascade_module()
        emitter = args.get_niagara_emitter_context()

        # choose the correct niagara module depending on the converted renderer
        if emitter.find_renderer("RibbonRenderer") is not None:
            script_name = "InitializeRibbon"
            script_asset = ueFxUtils.create_asset_data(Paths.script_initialize_ribbon)
            script_version = [1, 1]
        else:
            script_name = "InitializeParticle"
            script_asset = ueFxUtils.create_asset_data(Paths.script_initialize_particle)
            script_version = [1, 0]

        # find/add the module script for init'ing the location.
        script_args = ue.CreateScriptContextArgs(script_asset, script_version)
        initialize_script = emitter.find_or_add_module_script(
            script_name,
            script_args,
            ue.ScriptExecutionCategory.PARTICLE_SPAWN)

        # get all properties from the cascade initial location module
        # noinspection PyTypeChecker
        (start_location_distribution,
         distribute_over_n_points,
         distribute_threshold
         ) = ueFxUtils.get_particle_module_location_props(cascade_module)

        #  todo implement "choose only n particles" dynamic input

        # if distribute over n points is not 0 or 1, special case handle the start location distribution to be over an 
        # equispaced range.
        if distribute_over_n_points != 0.0 and distribute_over_n_points != 1.0:
            range_n_input = c2nUtils.create_script_input_random_range(0.0, distribute_over_n_points)
            n_input = ueFxUtils.create_script_input_int(distribute_over_n_points)

            script_asset = ueFxUtils.create_asset_data(Paths.di_divide_float)
            script_args = ue.CreateScriptContextArgs(script_asset)
            div_float_script = ueFxUtils.create_script_context(script_args)
            div_float_script.set_parameter("A", range_n_input)
            div_float_script.set_parameter("B", n_input)
            indexer_input = ueFxUtils.create_script_input_dynamic(div_float_script, ue.NiagaraScriptInputType.FLOAT)

            options = c2nUtils.DistributionConversionOptions()
            options.set_custom_indexer(indexer_input)
            position_input = c2nUtils.create_script_input_for_distribution(start_location_distribution, options)
        else:
            indexer_input = ueFxUtils.create_script_input_linked_parameter(
                "Emitter.LoopedAge",
                ue.NiagaraScriptInputType.FLOAT)
            options = c2nUtils.DistributionConversionOptions()
            options.set_custom_indexer(indexer_input)
            position_input = c2nUtils.create_script_input_for_distribution(start_location_distribution, options)

        # set the position.
        mode_input = ueFxUtils.create_script_input_enum(Paths.enum_niagara_position_initialization_mode, "Direct Set")
        initialize_script.set_parameter("Position Mode", mode_input)
        
        script_args = ue.CreateScriptContextArgs(ueFxUtils.create_asset_data(Paths.di_vec_to_pos))
        pos_vec_script = ueFxUtils.create_script_context(script_args)
        pos_vec_script.set_parameter("Input Position", position_input)
        pos_vec_input = ueFxUtils.create_script_input_dynamic(pos_vec_script, ue.NiagaraScriptInputType.POSITION)
        
        initialize_script.set_parameter("Position", pos_vec_input)
