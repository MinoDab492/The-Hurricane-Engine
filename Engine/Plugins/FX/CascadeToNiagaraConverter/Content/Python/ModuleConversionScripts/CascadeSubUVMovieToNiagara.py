from ModuleConverterInterface import ModuleConverterInterface
import unreal as ue
from unreal import FXConverterUtilitiesLibrary as ueFxUtils
import CascadeToNiagaraHelperMethods as c2nUtils
import Paths


class CascadeSubUVMovieConverter(ModuleConverterInterface):

    @classmethod
    def get_input_cascade_module(cls):
        return ue.ParticleModuleSubUVMovie

    @classmethod
    def convert(cls, args):
        cascade_module = args.get_cascade_module()
        emitter = args.get_niagara_emitter_context()
        
        # get all properties from the cascade sub uv movie module
        # noinspection PyTypeChecker
        (animation,
         subuv_index_distribution,
         b_use_real_time
         ) = ueFxUtils.get_particle_module_sub_uv_props(cascade_module)

        # noinspection PyTypeChecker
        (b_use_emitter_time,
         framerate_distribution,
         start_frame
         ) = ueFxUtils.get_particle_module_sub_uv_movie_props(cascade_module)

        # find/add the module script for sub uv animation
        script_asset = ueFxUtils.create_asset_data(Paths.script_subuv_animation_v2)
        script_args = ue.CreateScriptContextArgs(script_asset)
        subuv_script = emitter.find_or_add_module_script(
            "SubUVMovie",
            script_args,
            ue.ScriptExecutionCategory.PARTICLE_UPDATE)

        # set the subuv mode to infinite for equivalent behavior to subuv movie.
        script_input = ueFxUtils.create_script_input_enum(Paths.enum_niagara_subuv_lookup_mode, "Infinite")
        subuv_script.set_parameter("SubUV Animation Mode", script_input)

        # get the max value off of the sub uv distribution as this will be the 
        # number of frames
        (success,
         min_frame_val,
         max_frame_val
         ) = ueFxUtils.get_distribution_min_max_values(subuv_index_distribution)

        if success is True:
            end_frame = max_frame_val.x
        else:
            # set a sensible default and log that we couldn't resolve the frame
            # count
            end_frame = 1
            subuv_script.log(
                "Could not determine number of frames in uv sequence!",
                ue.NiagaraMessageSeverity.WARNING)

        # set the number of frames
        start_frame_input = ueFxUtils.create_script_input_int(start_frame - 1)
        subuv_script.set_parameter("Start Frame", start_frame_input)
        end_frame_input = ueFxUtils.create_script_input_int(end_frame)
        subuv_script.set_parameter("End Frame", end_frame_input)

        # set the play rate
        if b_use_real_time:
            subuv_script.log(
                "Failed to set \"Use Emitter Time\": Niagara does not support this mode!",
                ue.NiagaraMessageSeverity.ERROR)
            
            #  todo Divide particle age by world time dilation for the play rate input. Not implemented as Niagara does 
            #  not currently subsume world time dilation.
            pass
            
        options = c2nUtils.DistributionConversionOptions()
        if b_use_emitter_time:
            options.set_index_by_emitter_age()
        
        #   todo play rate is 4x too fast
        play_rate_input = c2nUtils.create_script_input_for_distribution(framerate_distribution, options)
        
        subuv_script.set_parameter("Sub UV Play Rate", play_rate_input)
