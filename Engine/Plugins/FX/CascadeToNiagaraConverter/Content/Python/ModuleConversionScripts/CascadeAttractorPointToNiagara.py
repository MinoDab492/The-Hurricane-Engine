from ModuleConverterInterface import ModuleConverterInterface
import unreal as ue
from unreal import FXConverterUtilitiesLibrary as ueFxUtils
import CascadeToNiagaraHelperMethods as c2nUtils
import Paths


class CascadePointAttractorConverter(ModuleConverterInterface):

    @classmethod
    def get_input_cascade_module(cls):
        return ue.ParticleModuleAttractorPoint

    @classmethod
    def convert(cls, args):
        cascade_module = args.get_cascade_module()
        emitter = args.get_niagara_emitter_context()
        
        # get all properties from the cascade point attractor module
        # noinspection PyTypeChecker
        (position_distribution, range_distribution, strength_distribution, 
         b_strength_by_distance, b_affects_base_velocity, b_override_velocity, 
         b_use_world_space_position, b_positive_x, b_positive_y, b_positive_z, 
         b_negative_x, b_negative_y, b_negative_z
         ) = ueFxUtils.get_particle_module_attractor_point_props(cascade_module)

        # find/add the niagara point attractor module
        script_asset = ueFxUtils.create_asset_data(Paths.script_point_attractor)
        script_args = ue.CreateScriptContextArgs(script_asset)
        point_attractor_script = emitter.find_or_add_module_script(
            "PointAttractor",
            script_args,
            ue.ScriptExecutionCategory.PARTICLE_UPDATE)

        # find/add the niagara solve forces and velocity module to resolve the dependency.
        script_asset = ueFxUtils.create_asset_data(Paths.script_solve_forces_and_velocity)
        script_args = ue.CreateScriptContextArgs(script_asset)
        emitter.find_or_add_module_script(
            "SolveForcesAndVelocity",
            script_args,
            ue.ScriptExecutionCategory.PARTICLE_UPDATE)

        # set the attractor position
        position_input = c2nUtils.create_script_input_for_distribution(position_distribution)
        point_attractor_script.set_parameter("AttractorPosition", position_input)
        
        # set the attraction strength
        strength_input = c2nUtils.create_script_input_for_distribution(strength_distribution)
        point_attractor_script.set_parameter("AttractionStrength", strength_input)

        # set the attraction range (radius)
        range_input = c2nUtils.create_script_input_for_distribution(range_distribution)
        point_attractor_script.set_parameter("Attraction Radius", range_input)
        
        # if strength by distance is specified, enable that for the module.
        falloff_input = ueFxUtils.create_script_input_float(0.5)
        if b_strength_by_distance:
            point_attractor_script.set_parameter("Falloff Exponent", falloff_input, True, True)
        else:
            # set the enabled state for falloff to OFF
            point_attractor_script.set_parameter("Falloff Exponent", falloff_input, True, False)
            
            # todo add module support for cascade parity

