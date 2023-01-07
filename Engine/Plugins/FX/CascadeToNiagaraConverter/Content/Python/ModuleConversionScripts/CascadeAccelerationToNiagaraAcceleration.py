from ModuleConverterInterface import ModuleConverterInterface
import unreal as ue
from unreal import FXConverterUtilitiesLibrary as ueFxUtils
import CascadeToNiagaraHelperMethods as c2nUtils
import Paths


class CascadeAccelerationConverter(ModuleConverterInterface):

    @classmethod
    def get_input_cascade_module(cls):
        return ue.ParticleModuleAcceleration

    @classmethod
    def convert(cls, args):
        cascade_module = args.get_cascade_module()
        emitter = args.get_niagara_emitter_context()

        # find/add the module script for acceleration force
        script_asset = ueFxUtils.create_asset_data(Paths.script_acceleration_force)
        script_args = ue.CreateScriptContextArgs(script_asset)
        acceleration_script = emitter.find_or_add_module_script(
            "Acceleration",
            script_args,
            ue.ScriptExecutionCategory.PARTICLE_UPDATE)

        # get all properties from the cascade acceleration module
        # noinspection PyTypeChecker
        (acceleration_distribution,
         b_apply_owner_scale
         ) = ueFxUtils.get_particle_module_acceleration_props(cascade_module)

        # log that we are skipping apply owner scale for now
        if b_apply_owner_scale:
            acceleration_script.log(
                "Skipped converting b_apply_owner_scale of cascade acceleration module; niagara equivalent module does "
                "not support this mode!",
                ue.NiagaraMessageSeverity.WARNING)
        else:
            pass

        # make an input to apply the acceleration vector
        acceleration_input = c2nUtils.create_script_input_for_distribution(acceleration_distribution)

        # cache the acceleration value in the spawn script
        emitter.set_parameter_directly(
            "Particles.Acceleration",
            acceleration_input,
            ue.ScriptExecutionCategory.PARTICLE_SPAWN)

        # set the acceleration value
        cached_acceleration_input = ueFxUtils.create_script_input_linked_parameter(
            "Particles.Acceleration",
            ue.NiagaraScriptInputType.VEC3)
        acceleration_script.set_parameter("Acceleration", cached_acceleration_input)

        # make sure there is a solve forces and velocity module.
        script_asset = ueFxUtils.create_asset_data(Paths.script_solve_forces_and_velocity)
        script_args = ue.CreateScriptContextArgs(script_asset)
        emitter.find_or_add_module_script(
            "SolveForcesAndVelocity",
            script_args,
            ue.ScriptExecutionCategory.PARTICLE_UPDATE)
