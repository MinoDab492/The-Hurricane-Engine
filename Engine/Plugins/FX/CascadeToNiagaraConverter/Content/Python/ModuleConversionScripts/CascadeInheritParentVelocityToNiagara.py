from ModuleConverterInterface import ModuleConverterInterface
import unreal as ue
from unreal import FXConverterUtilitiesLibrary as ueFxUtils
import CascadeToNiagaraHelperMethods as c2nUtils
import Paths


class CascadeInheritParentVelocityConverter(ModuleConverterInterface):

    @classmethod
    def get_input_cascade_module(cls):
        return ue.ParticleModuleVelocityInheritParent

    @classmethod
    def convert(cls, args):
        cascade_module = args.get_cascade_module()
        emitter = args.get_niagara_emitter_context()

        # get all properties of the cascade inherit parent velocity module.
        # noinspection PyTypeChecker
        (scale_distribution,
         b_world_space,
         b_apply_owner_scale
         ) = ueFxUtils.get_particle_module_velocity_inherit_parent_props(cascade_module)

        # find/add the inherit velocity module.
        script_asset = ueFxUtils.create_asset_data(Paths.script_inherit_parent_velocity)
        script_args = ue.CreateScriptContextArgs(script_asset, [1, 1])
        inherit_vel_script = emitter.find_or_add_module_script(
            "InheritVelocity",
            script_args,
            ue.ScriptExecutionCategory.PARTICLE_SPAWN)

        #  todo consider apply initial forces as well

        # find/add the module script for solving forces and velocity
        script_asset = ueFxUtils.create_asset_data(Paths.script_solve_forces_and_velocity)
        script_args = ue.CreateScriptContextArgs(script_asset)
        emitter.find_or_add_module_script(
            "SolveForcesAndVelocity",
            script_args,
            ue.ScriptExecutionCategory.PARTICLE_UPDATE)

        # make an input for the velocity scale.
        options = c2nUtils.DistributionConversionOptions()
        options.set_custom_indexer(
            ueFxUtils.create_script_input_linked_parameter("Emitter.LoopedAge", ue.NiagaraScriptInputType.FLOAT))
        scale_vel_input = c2nUtils.create_script_input_for_distribution(
            scale_distribution,
            options)

        #  todo can we skip world space flag

        # apply owner scale if required.
        if b_apply_owner_scale:
            script_asset = ueFxUtils.create_asset_data(Paths.di_multiply_vector)
            script_args = ue.CreateScriptContextArgs(script_asset)
            mul_vec_script = ueFxUtils.create_script_context(script_args)
            mul_vec_script.set_parameter("A", scale_vel_input)
            mul_vec_script.set_parameter("B", ueFxUtils.create_script_input_linked_parameter(
                "Engine.Owner.Scale",
                 ue.NiagaraScriptInputType.VEC3))
            scale_vel_input = ueFxUtils.create_script_input_dynamic(
                mul_vec_script,
                ue.NiagaraScriptInputType.VEC3)

        # set the velocity scale.
        inherit_vel_script.set_parameter("Inherited Velocity Amount Scale", scale_vel_input)

        # disable the velocity limit, cascade does not enforce this.
        inherit_vel_script.set_parameter(
            "Inherited Velocity Speed Limit",
            ueFxUtils.create_script_input_float(0.0),
            True,
            False)
