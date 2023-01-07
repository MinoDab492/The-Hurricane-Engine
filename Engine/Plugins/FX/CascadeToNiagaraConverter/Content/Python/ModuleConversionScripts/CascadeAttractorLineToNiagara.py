from ModuleConverterInterface import ModuleConverterInterface
import unreal as ue
from unreal import FXConverterUtilitiesLibrary as ueFxUtils
import CascadeToNiagaraHelperMethods as c2nUtils
import Paths


class CascadeLineAttractorConverter(ModuleConverterInterface):

    @classmethod
    def get_input_cascade_module(cls):
        return ue.ParticleModuleAttractorLine

    @classmethod
    def convert(cls, args):
        cascade_module = args.get_cascade_module()
        emitter = args.get_niagara_emitter_context()
        
        # get all properties from the cascade line attractor module
        # noinspection PyTypeChecker
        (start_point,
         end_point,
         range_distribution,
         strength_distribution
         ) = ueFxUtils.get_particle_module_attractor_line_props(cascade_module)
        
        # find/add the niagara line attractor module
        script_asset = ueFxUtils.create_asset_data(Paths.script_line_attractor)
        script_args = ue.CreateScriptContextArgs(script_asset)
        line_attractor_script = emitter.find_or_add_module_script(
            "LineAttractor",
            script_args,
            ue.ScriptExecutionCategory.PARTICLE_UPDATE)
        
        # find/add the niagara solve forces and velocity module to resolve the dependency.
        script_asset = ueFxUtils.create_asset_data(Paths.script_solve_forces_and_velocity)
        script_args = ue.CreateScriptContextArgs(script_asset)
        emitter.find_or_add_module_script(
            "SolveForcesAndVelocity",
            script_args,
            ue.ScriptExecutionCategory.PARTICLE_UPDATE)
        
        # set the start and end points
        start_point_input = ueFxUtils.create_script_input_vector(start_point)
        end_point_input = ueFxUtils.create_script_input_vector(end_point)
        line_attractor_script.set_parameter("Line Start", start_point_input)
        line_attractor_script.set_parameter("Line End", end_point_input)
        
        # set the attraction strength
        strength_input = c2nUtils.create_script_input_for_distribution(strength_distribution)
        line_attractor_script.set_parameter("Attraction Strength", strength_input)
        
        # set the attraction range
        range_input = c2nUtils.create_script_input_for_distribution(range_distribution)
        line_attractor_script.set_parameter("Attraction Falloff", range_input, True, True)
