from ModuleConverterInterface import ModuleConverterInterface
import unreal as ue
from unreal import FXConverterUtilitiesLibrary as ueFxUtils
import CascadeToNiagaraHelperMethods as c2nUtils
import Paths


class CascadeMeshRotationRateConverter(ModuleConverterInterface):

    @classmethod
    def get_input_cascade_module(cls):
        return ue.ParticleModuleMeshRotationRate

    @classmethod
    def convert(cls, args):
        cascade_module = args.get_cascade_module()
        emitter = args.get_niagara_emitter_context()
        
        # get all properties of the mesh rotation rate module.
        # noinspection PyTypeChecker
        start_rotation_rate_distribution = ueFxUtils.get_particle_module_mesh_rotation_rate_props(cascade_module)
        
        # add the update mesh orientation script.
        script_asset = ueFxUtils.create_asset_data(Paths.script_update_mesh_orientation)
        script_args = ue.CreateScriptContextArgs(script_asset, [1, 1])
        mesh_orient_script = emitter.find_or_add_module_script(
            "UpdateMeshOrientation",
            script_args,
            ue.ScriptExecutionCategory.PARTICLE_UPDATE)
        
        # evaluate the rotation vector in spawn, and set it on the script in 
        # update.
        options = c2nUtils.DistributionConversionOptions()
        emitter_age_index_input = ueFxUtils.create_script_input_linked_parameter(
            "Emitter.LoopedAge",
            ue.NiagaraScriptInputType.FLOAT)
        
        options.set_custom_indexer(emitter_age_index_input)
        rotation_vec_input = c2nUtils.create_script_input_for_distribution(start_rotation_rate_distribution, options)
        
        emitter.set_parameter_directly(
            "Particles.SpawnRotationVector",
            rotation_vec_input,
            ue.ScriptExecutionCategory.PARTICLE_SPAWN)
        
        spawn_rotation_vec_input = ueFxUtils.create_script_input_linked_parameter(
            "Particles.SpawnRotationVector",
            ue.NiagaraScriptInputType.VEC3)
        
        mesh_orient_script.set_parameter("Rotation Vector", spawn_rotation_vec_input)
