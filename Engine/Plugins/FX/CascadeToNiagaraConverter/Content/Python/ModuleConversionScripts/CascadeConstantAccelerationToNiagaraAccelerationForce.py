from ModuleConverterInterface import ModuleConverterInterface
import unreal as ue
from unreal import FXConverterUtilitiesLibrary as ueFxUtils
import Paths


class CascadeConstantAccelerationConverter(ModuleConverterInterface):

    @classmethod
    def get_input_cascade_module(cls):
        return ue.ParticleModuleAccelerationConstant

    @classmethod
    def convert(cls, args):
        cascade_module = args.get_cascade_module()
        emitter = args.get_niagara_emitter_context()

        # find/add the module script for acceleration force
        script_asset = ueFxUtils.create_asset_data(Paths.script_acceleration_force)
        script_args = ue.CreateScriptContextArgs(script_asset)
        acceleration_force_script = emitter.find_or_add_module_script(
            "ConstantAccelerationForce",
            script_args,
            ue.ScriptExecutionCategory.PARTICLE_UPDATE)

        # get all properties from the cascade constant acceleration module
        # noinspection PyTypeChecker
        acceleration_vec = ueFxUtils.get_particle_module_constant_acceleration_props(cascade_module)

        # make an input to apply the acceleration vector
        acceleration_input = ueFxUtils.create_script_input_vector(acceleration_vec)

        # set the acceleration vector value
        acceleration_force_script.set_parameter("Acceleration", acceleration_input)

        # make sure there is a solve forces and velocity module.
        script_asset = ueFxUtils.create_asset_data(Paths.script_solve_forces_and_velocity)
        script_args = ue.CreateScriptContextArgs(script_asset)
        emitter.find_or_add_module_script(
            "SolveForcesAndVelocity",
            script_args,
            ue.ScriptExecutionCategory.PARTICLE_UPDATE)
