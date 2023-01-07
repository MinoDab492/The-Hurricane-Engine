from ModuleConverterInterface import ModuleConverterInterface
import unreal as ue
from unreal import FXConverterUtilitiesLibrary as ueFxUtils
import CascadeToNiagaraHelperMethods as c2nUtils
import Paths


class CascadeVelocityByLifeConverter(ModuleConverterInterface):

    @classmethod
    def get_input_cascade_module(cls):
        return ue.ParticleModuleVelocityOverLifetime

    @classmethod
    def convert(cls, args):
        cascade_module = args.get_cascade_module()
        emitter = args.get_niagara_emitter_context()

        # get velocity over lifetime properties.
        # noinspection PyTypeChecker
        (vel_over_life_distribution,
         b_absolute,
         b_world_space,
         b_apply_owner_scale
         ) = ueFxUtils.get_particle_module_velocity_over_lifetime_props(cascade_module)

        # make sure there is a solve forces and velocity module.
        script_asset = ueFxUtils.create_asset_data(Paths.script_solve_forces_and_velocity)
        script_args = ue.CreateScriptContextArgs(script_asset)
        emitter.find_or_add_module_script(
            "SolveForcesAndVelocity",
            script_args,
            ue.ScriptExecutionCategory.PARTICLE_UPDATE)

        # make input for velocity over life.
        vel_over_life_input = c2nUtils.create_script_input_for_distribution(
            vel_over_life_distribution)

        # apply owner scale if necessary.
        if b_apply_owner_scale:
            script_asset = ueFxUtils.create_asset_data(Paths.di_multiply_vector)
            script_args = ue.CreateScriptContextArgs(script_asset)
            mul_vec_script = ueFxUtils.create_script_context(script_args)
            
            owner_scale_input = ueFxUtils.create_script_input_linked_parameter(
                "Engine.Owner.Scale",
                ue.NiagaraScriptInputType.VEC3)

            mul_vec_script.set_parameter("A", vel_over_life_input)
            mul_vec_script.set_parameter("B", owner_scale_input)

            vel_over_life_input = ueFxUtils.create_script_input_dynamic(
                mul_vec_script,
                ue.NiagaraScriptInputType.VEC3)

        # choose behavior based on b_absolute.
        # if True, set velocity directly. 
        # if False, scale velocity.
        if b_absolute:
            emitter.set_parameter_directly(
                "Particles.Velocity",
                vel_over_life_input,
                ue.ScriptExecutionCategory.PARTICLE_UPDATE)
        else:
            script_asset = ueFxUtils.create_asset_data(Paths.script_scale_velocity)
            script_args = ue.CreateScriptContextArgs(script_asset)
            scale_velocity_script = emitter.find_or_add_module_script(
                "ScaleVelocity",
                script_args,
                ue.ScriptExecutionCategory.PARTICLE_UPDATE)
            scale_velocity_script.set_parameter(
                "Velocity Scale",
                vel_over_life_input)

            # use appropriate coordinate space.
            if b_world_space:
                coordinate_space_name = "World"
            else:
                coordinate_space_name = "Local"

            script_input = ueFxUtils.create_script_input_enum(
                Paths.enum_niagara_coordinate_space,
                coordinate_space_name)
            scale_velocity_script.set_parameter(
                "Coordinate Space",
                script_input)
