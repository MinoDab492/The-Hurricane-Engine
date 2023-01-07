from ModuleConverterInterface import ModuleConverterInterface
import unreal as ue
from unreal import FXConverterUtilitiesLibrary as ueFxUtils
import CascadeToNiagaraHelperMethods as c2nUtils
import Paths


class CascadeParticleLifetimeConverter(ModuleConverterInterface):

    @classmethod
    def get_input_cascade_module(cls):
        return ue.ParticleModuleLifetime

    @classmethod
    def convert(cls, args):
        cascade_module = args.get_cascade_module()
        emitter = args.get_niagara_emitter_context()

        # find/add the module script for particle state
        script_args = ue.CreateScriptContextArgs(ueFxUtils.create_asset_data(Paths.script_particle_state), [1, 1])
        particle_state_script = emitter.find_or_add_module_script(
            "ParticleState",
            script_args,
            ue.ScriptExecutionCategory.PARTICLE_UPDATE)
        
        # get all properties from the cascade particle lifetime module
        # noinspection PyTypeChecker
        lifetime_distribution = ueFxUtils.get_particle_module_lifetime_props(cascade_module)
        if c2nUtils.distribution_always_equals(lifetime_distribution, 0.0):
            # early exit if lifetime is set to 0; special case for infinite 
            # lifetime
            particle_state_script.set_parameter(
                "Kill Particles When Lifetime Has Elapsed",
                ueFxUtils.create_script_input_bool(False))
            particle_state_script.set_parameter(
                "Let Infinitely Lived Particles Die When Emitter Deactivates",
                ueFxUtils.create_script_input_bool(True))
            return
        
        # find/add the module script for init particle
        if emitter.find_renderer("RibbonRenderer") is not None:
            script_name = "InitializeRibbon"
            initialize_script_asset_data = ueFxUtils.create_asset_data(Paths.script_initialize_ribbon)
            initialize_script_version = [1, 1]
        elif emitter.find_renderer("MeshRenderer") is not None:
            script_name = "InitializeParticle"
            initialize_script_asset_data = ueFxUtils.create_asset_data(Paths.script_initialize_particle)
            initialize_script_version = [1, 0]
        else:
            script_name = "InitializeParticle"
            initialize_script_asset_data = ueFxUtils.create_asset_data(Paths.script_initialize_particle)
            initialize_script_version = [1, 0]
           
        script_args = ue.CreateScriptContextArgs(initialize_script_asset_data, initialize_script_version) 
        initialize_script = emitter.find_or_add_module_script(
            script_name,
            script_args,
            ue.ScriptExecutionCategory.PARTICLE_SPAWN)

        # convert the lifetime property
        lifetime_input = c2nUtils.create_script_input_for_distribution(lifetime_distribution)

        # set the lifetime
        initialize_script.set_parameter("Lifetime", lifetime_input)
