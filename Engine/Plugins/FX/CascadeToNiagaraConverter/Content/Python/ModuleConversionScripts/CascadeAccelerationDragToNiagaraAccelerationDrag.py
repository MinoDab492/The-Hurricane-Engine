from ModuleConverterInterface import ModuleConverterInterface
import unreal as ue
from unreal import FXConverterUtilitiesLibrary as ueFxUtils
import CascadeToNiagaraHelperMethods as c2nUtils
import Paths


class CascadeAccelerationDragConverter(ModuleConverterInterface):

    @classmethod
    def get_input_cascade_module(cls):
        return ue.ParticleModuleAccelerationDrag

    @classmethod
    def convert(cls, args):
        cascade_module = args.get_cascade_module()
        emitter = args.get_niagara_emitter_context()

        # find/add the module script for drag
        script_asset = ueFxUtils.create_asset_data(Paths.script_drag)
        script_args = ue.CreateScriptContextArgs(script_asset, [1, 1])
        drag_script = emitter.find_or_add_module_script(
            "Drag", 
            script_args, 
            ue.ScriptExecutionCategory.PARTICLE_UPDATE)

        # get all properties from the cascade acceleration module
        # noinspection PyTypeChecker
        acceleration_drag = ueFxUtils.get_particle_module_acceleration_drag_props(cascade_module)

        # make an input to apply the acceleration drag
        acceleration_drag_input = c2nUtils.create_script_input_for_distribution(acceleration_drag)

        # set the acceleration value
        drag_script.set_parameter("Drag", acceleration_drag_input)

        # make sure there is a solve forces and velocity module.
        script_asset = ueFxUtils.create_asset_data(Paths.script_solve_forces_and_velocity)
        script_args = ue.CreateScriptContextArgs(script_asset, [1, 0])
        emitter.find_or_add_module_script(
            "SolveForcesAndVelocity",
            script_args,
            ue.ScriptExecutionCategory.PARTICLE_UPDATE)
