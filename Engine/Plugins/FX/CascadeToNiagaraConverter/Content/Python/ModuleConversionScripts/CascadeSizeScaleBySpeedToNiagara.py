from ModuleConverterInterface import ModuleConverterInterface
import unreal as ue
from unreal import FXConverterUtilitiesLibrary as ueFxUtils
import Paths


class CascadeSizeScaleBySpeedConverter(ModuleConverterInterface):

    @classmethod
    def get_input_cascade_module(cls):
        return ue.ParticleModuleSizeScaleBySpeed

    @classmethod
    def convert(cls, args):
        cascade_module = args.get_cascade_module()
        emitter = args.get_niagara_emitter_context()

        # get all properties from the cascade size scale by speed module
        # noinspection PyTypeChecker
        (speed_scale,
         max_scale
         ) = ueFxUtils.get_particle_module_size_scale_by_speed_props(cascade_module)

        # fudge max scale by 10x to get equivalent result for niagara
        max_scale = max_scale * 10.0
        
        # make vector for velocity scale
        velocity_scale = ue.Vector(speed_scale.x, speed_scale.y, 1.0)

        # multiply the source velocity by the scalar
        script_asset = ueFxUtils.create_asset_data(Paths.di_multiply_vector)
        script_args = ue.CreateScriptContextArgs(script_asset)
        multiply_vector_script = ueFxUtils.create_script_context(script_args)
        velocity_scale_input = ueFxUtils.create_script_input_vector(velocity_scale)
        velocity_input = ueFxUtils.create_script_input_linked_parameter(
            "Particles.Velocity",
            ue.NiagaraScriptInputType.VEC3)
        multiply_vector_script.set_parameter("A", velocity_scale_input)
        multiply_vector_script.set_parameter("B", velocity_input)
        scaled_velocity_input = ueFxUtils.create_script_input_dynamic(
            multiply_vector_script,
            ue.NiagaraScriptInputType.VEC3)

        # choose the module to use for this scaling
        if emitter.find_renderer("RibbonRenderer") is not None:
            emitter.log(
                "Skipped converting Cascade module SizeScaleBySpeed: converter does not support SizeScaleBySpeed for "
                "Niagara ribbons.",
                ue.NiagaraMessageSeverity.ERROR)
            return
        elif emitter.find_renderer("MeshRenderer") is not None:
            # use the scale mesh by speed script for mesh renderer
            script_asset = ueFxUtils.create_asset_data(Paths.script_scale_mesh_size_speed)
            script_args = ue.CreateScriptContextArgs(script_asset)
            scale_mesh_script = emitter.find_or_add_module_script(
                "ScaleMeshSizeBySpeed",
                script_args,
                ue.ScriptExecutionCategory.PARTICLE_UPDATE)

            # set the scaled velocity
            scale_mesh_script.set_parameter("Source Velocity",  scaled_velocity_input)

            # make vector for the max scale
            max_scale = ue.Vector(max_scale.x, max_scale.y, 1.0)

            # set the max scale
            max_scale_input = ueFxUtils.create_script_input_vector(max_scale)
            scale_mesh_script.set_parameter("Max Scale Factor", max_scale_input)
            
        else:
            # use the scale sprite by speed script for sprite renderer
            script_asset = ueFxUtils.create_asset_data(Paths.script_scale_sprite_size_speed)
            script_args = ue.CreateScriptContextArgs(script_asset)
            scale_sprite_script = emitter.find_or_add_module_script(
                "ScaleSpriteSizeBySpeed",
                script_args,
                ue.ScriptExecutionCategory.PARTICLE_UPDATE)

            # set the scaled velocity
            scale_sprite_script.set_parameter("Source Velocity", scaled_velocity_input)

            # make vector for the max scale
            max_scale = ue.Vector2D(max_scale.x, max_scale.y)
            
            # set the max scale
            max_scale_input = ueFxUtils.create_script_input_vec2(max_scale)
            scale_sprite_script.set_parameter("Max Scale Factor", max_scale_input)
