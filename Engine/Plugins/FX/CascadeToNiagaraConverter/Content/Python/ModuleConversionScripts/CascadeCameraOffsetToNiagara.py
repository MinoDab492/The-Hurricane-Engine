from ModuleConverterInterface import ModuleConverterInterface
import unreal as ue
from unreal import FXConverterUtilitiesLibrary as ueFxUtils
import CascadeToNiagaraHelperMethods as c2nUtils
import Paths


class CascadeCameraOffsetConverter(ModuleConverterInterface):

    @classmethod
    def get_input_cascade_module(cls):
        return ue.ParticleModuleCameraOffset

    @classmethod
    def convert(cls, args):
        cascade_module = args.get_cascade_module()
        emitter = args.get_niagara_emitter_context()
        
        # get all properties from the cascade camera offset module
        # noinspection PyTypeChecker
        (camera_offset_distribution,
         b_spawn_time_only,
         update_method
         ) = ueFxUtils.get_particle_module_camera_offset_props(cascade_module)
        
        # find/add the module script for camera offset, choose spawn or update
        # based on the cascade module
        script_asset = ueFxUtils.create_asset_data(Paths.script_camera_offset)
        script_args = ue.CreateScriptContextArgs(script_asset)
        if b_spawn_time_only is True:
            camera_offset_script = emitter.find_or_add_module_script(
                "CameraOffset",
                script_args,
                ue.ScriptExecutionCategory.PARTICLE_SPAWN)
        else:
            camera_offset_script = emitter.find_or_add_module_script(
                "CameraOffset",
                script_args,
                ue.ScriptExecutionCategory.PARTICLE_UPDATE)
        
        # make an input to apply the camera offset
        offset_input = c2nUtils.create_script_input_for_distribution(camera_offset_distribution)
        
        # apply the input as directed by the cascade module update method
        if update_method == ue.ParticleCameraOffsetUpdateMethod.EPCOUM_DIRECT_SET:
            camera_offset_script.set_parameter("Camera Offset Amount", offset_input)
            
        elif update_method == ue.ParticleCameraOffsetUpdateMethod.EPCOUM_ADDITIVE:
            script_asset = ueFxUtils.create_asset_data(Paths.di_add_floats)
            script_args = ue.CreateScriptContextArgs(script_asset)
            add_float_script = ueFxUtils.create_script_context(script_args)
                
            original_offset_input = ueFxUtils.create_script_input_linked_parameter(
                "Particles.CameraOffset", ue.NiagaraScriptInputType.FLOAT)
            add_float_script.set_parameter("A", original_offset_input)
            add_float_script.set_parameter("B", offset_input)
            final_offset_input = ueFxUtils.create_script_input_dynamic(
                add_float_script,
                ue.NiagaraScriptInputType.FLOAT)
            camera_offset_script.set_parameter("Camera Offset Amount", final_offset_input)
            
        elif update_method == ue.ParticleCameraOffsetUpdateMethod.EPCOUM_SCALAR:
            script_asset = ueFxUtils.create_asset_data(Paths.di_multiply_float)
            create_script_args = ue.CreateScriptContextArgs(script_asset)
            multiply_float_script = ueFxUtils.create_script_context(create_script_args)
            original_offset_input = ueFxUtils.create_script_input_linked_parameter(
                "Particles.CameraOffset",
                ue.NiagaraScriptInputType.FLOAT)
            multiply_float_script.set_parameter("A", original_offset_input)
            multiply_float_script.set_parameter("B", offset_input)
            final_offset_input = ueFxUtils.create_script_input_dynamic(
                multiply_float_script,
                ue.NiagaraScriptInputType.FLOAT)
            camera_offset_script.set_parameter("Camera Offset Amount", final_offset_input)
            
        else:
            camera_offset_script.log(
                "Encountered unknown particle camera offset update method when converting cascade camera offset!",
                ue.NiagaraMessageSeverity.ERROR)
