from ModuleConverterInterface import ModuleConverterInterface
import unreal as ue
from unreal import FXConverterUtilitiesLibrary as ueFxUtils
import CascadeToNiagaraHelperMethods as c2nUtils
import Paths


class CascadeParticleVelocityConverter(ModuleConverterInterface):

    @classmethod
    def get_input_cascade_module(cls):
        return ue.ParticleModuleVelocity

    @classmethod
    def convert(cls, args):
        cascade_module = args.get_cascade_module()
        emitter = args.get_niagara_emitter_context()

        # get all properties from the cascade particle velocity module
        # noinspection PyTypeChecker
        (start_velocity_distribution,
         start_velocity_radial_distribution,
         b_world_space,
         b_apply_owner_scale
         ) = ueFxUtils.get_particle_module_velocity_props(cascade_module)

        b_need_solver = False
        if c2nUtils.distribution_always_equals(start_velocity_distribution, 0.0) is False:
            b_need_solver = True

            # find/add the module script for adding velocity in a cone
            script_asset = ueFxUtils.create_asset_data(Paths.script_add_velocity)
            script_args = ue.CreateScriptContextArgs(script_asset, [1, 2])
            add_velocity_script = emitter.find_or_add_module_script(
                "AddVelocityInCone",
                script_args,
                ue.ScriptExecutionCategory.PARTICLE_SPAWN)

            # convert the distribution properties
            start_velocity_input = c2nUtils.create_script_input_for_distribution(
                start_velocity_distribution)

            # set the start velocity
            add_velocity_script.set_parameter("Velocity", start_velocity_input)

        if c2nUtils.distribution_always_equals(
            start_velocity_radial_distribution,
            0.0
        ) is False:
            b_need_solver = True

            # find/add the module script for adding force from a point
            script_asset = ueFxUtils.create_asset_data(Paths.script_add_velocity)
            script_args = ue.CreateScriptContextArgs(script_asset, [1, 2])
            add_velocity_point_script = emitter.find_or_add_module_script(
                "PointForce",
                script_args,
                ue.ScriptExecutionCategory.PARTICLE_SPAWN)

            # convert the distribution properties
            start_velocity_radial_input = c2nUtils.create_script_input_for_distribution(
                start_velocity_radial_distribution)

            # set the point velocity
            add_velocity_point_script.set_parameter(
                "Velocity Strength",
                start_velocity_radial_input)

        if b_need_solver:
            # find/add the module script for solving forces and velocity
            script_asset = ueFxUtils.create_asset_data(Paths.script_solve_forces_and_velocity)
            script_args = ue.CreateScriptContextArgs(script_asset)
            emitter.find_or_add_module_script(
                "SolveForcesAndVelocity",
                script_args,
                ue.ScriptExecutionCategory.PARTICLE_UPDATE)
