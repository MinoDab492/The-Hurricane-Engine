from ModuleConverterInterface import ModuleConverterInterface
import unreal as ue
from unreal import FXConverterUtilitiesLibrary as ueFxUtils
import CascadeToNiagaraHelperMethods as c2nUtils
import Paths


class CascadeParticleSizeScaleConverter(ModuleConverterInterface):

    @classmethod
    def get_input_cascade_module(cls):
        return ue.ParticleModuleSizeScale

    @classmethod
    def convert(cls, args):
        cascade_module = args.get_cascade_module()
        emitter = args.get_niagara_emitter_context()
        
        # get all properties of the size scale module
        # noinspection PyTypeChecker
        size_scale_distribution = ueFxUtils.get_particle_module_size_scale_props(cascade_module)
        
        options = c2nUtils.DistributionConversionOptions()
        # choose the correct parameter and conversion options depending on renderer
        if emitter.find_renderer("RibbonRenderer") is not None:
            options.set_target_type_width(ue.NiagaraScriptInputType.FLOAT)
            scale_input = c2nUtils.create_script_input_for_distribution(size_scale_distribution, options)
            
            script_args = ue.CreateScriptContextArgs(ueFxUtils.create_asset_data(Paths.script_scale_ribbon_width), [1, 0])
            script = emitter.find_or_add_module_script(
                "ScaleRibbonWidth",
                script_args,
                ue.ScriptExecutionCategory.PARTICLE_UPDATE)
            
            script.set_parameter("Ribbon Width Scale", scale_input)
            
        elif emitter.find_renderer("MeshRenderer") is not None:
            options.set_target_type_width(ue.NiagaraScriptInputType.VEC3)
            scale_input = c2nUtils.create_script_input_for_distribution(size_scale_distribution, options)
            
            script_args = ue.CreateScriptContextArgs(ueFxUtils.create_asset_data(Paths.script_scale_mesh_size), [1, 0])
            script = emitter.find_or_add_module_script(
                "ScaleMeshSize",
                script_args,
                ue.ScriptExecutionCategory.PARTICLE_UPDATE)
            
            script.set_parameter("Scale Factor", scale_input)
            
        else:
            options.set_target_type_width(ue.NiagaraScriptInputType.FLOAT)
            scale_input = c2nUtils.create_script_input_for_distribution(size_scale_distribution, options)
            scale_mode_input = ueFxUtils.create_script_input_enum(Paths.enum_niagara_scale_sprite_size, "Uniform")

            script_args = ue.CreateScriptContextArgs(ueFxUtils.create_asset_data(Paths.script_scale_sprite_size), [1, 1])
            script = emitter.find_or_add_module_script(
                "ScaleSpriteSize",
                script_args,
                ue.ScriptExecutionCategory.PARTICLE_UPDATE)

            script.set_parameter("Scale Sprite Size Mode", scale_mode_input)
            script.set_parameter("Uniform Scale Factor", scale_input)

        # find/add the module script for particle state
        script_args = ue.CreateScriptContextArgs(ueFxUtils.create_asset_data(Paths.script_particle_state), [1, 1])
        emitter.find_or_add_module_script("ParticleState", script_args, ue.ScriptExecutionCategory.PARTICLE_UPDATE)
