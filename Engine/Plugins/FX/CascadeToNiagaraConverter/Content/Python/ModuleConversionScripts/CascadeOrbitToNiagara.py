from ModuleConverterInterface import ModuleConverterInterface
import unreal as ue
from unreal import FXConverterUtilitiesLibrary as ueFxUtils
import CascadeToNiagaraHelperMethods as c2nUtils
import Paths


class CascadeOrbitConverter(ModuleConverterInterface):
    # mark when the first link mode orbit module is visited.
    b_first_link_visited = False

    @classmethod
    def get_input_cascade_module(cls):
        return ue.ParticleModuleOrbit

    @classmethod
    def convert(cls, args):
        cascade_module = args.get_cascade_module()
        emitter = args.get_niagara_emitter_context()

        # get all properties of the orbit module.
        # noinspection PyTypeChecker
        (chain_mode,
         offset_amount_distribution,
         offset_options,
         rotation_distribution,
         rotation_options,
         rotation_rate_distribution,
         rotation_rate_options
         ) = ueFxUtils.get_particle_module_orbit_props(cascade_module)

        # add a converter specific orbit script to the emitter.
        script_asset = ueFxUtils.create_asset_data(Paths.script_orbit)
        script_args = ue.CreateScriptContextArgs(script_asset)
        orbit_script = ueFxUtils.create_script_context(script_args)
        emitter.add_module_script(
            "Orbit",
            orbit_script,
            ue.ScriptExecutionCategory.PARTICLE_UPDATE)

        # make sure there is also a solver script.
        script_asset = ueFxUtils.create_asset_data(Paths.script_solve_orbit)
        script_args = ue.CreateScriptContextArgs(script_asset)
        emitter.find_or_add_module_script(
            "SolveOrbit",
            script_args,
            ue.ScriptExecutionCategory.PARTICLE_UPDATE)

        # point the renderers position binding to the output position of the 
        # orbit solver script.
        renderers = emitter.get_all_renderers()
        for renderer in renderers:
            # noinspection PyTypeChecker
            emitter.set_renderer_binding(
                renderer,
                "PositionBinding",
                "Particles.OrbitOffsetPosition",
                ue.NiagaraRendererSourceDataMode.PARTICLES)

        # choose behavior based on the orbit chain mode.
        b_check_first_orbit_link = False
        if chain_mode == ue.OrbitChainMode.EO_CHAIN_MODE_ADD:
            enum_value_name = "Add"
        elif chain_mode == ue.OrbitChainMode.EO_CHAIN_MODE_LINK:
            enum_value_name = "Link"
            b_check_first_orbit_link = True
        elif chain_mode == ue.OrbitChainMode.EO_CHAIN_MODE_SCALE:
            enum_value_name = "Scale"
        else:
            raise NameError("Encountered invalid chain mode when converting "
                            "cascade orbit module!")
        orbit_script.set_parameter(
            "Chain Mode",
            ueFxUtils.create_script_input_enum(Paths.enum_cascade_niagara_orbit_mode, enum_value_name))

        # set the static switch if this is the first link mode orbit module.
        if cls.b_first_link_visited is False and b_check_first_orbit_link:
            cls.b_first_link_visited = True
            orbit_script.set_parameter("First Chain Mode Link", ueFxUtils.create_script_input_bool(True))

        def set_orbit_param(distribution, options, param_name):
            convert_options = c2nUtils.DistributionConversionOptions()
            if options.get_editor_property("bProcessDuringSpawn"):
                convert_options.set_b_evaluate_spawn_only(True)
            elif options.get_editor_property("bProcessDuringUpdate"):
                convert_options.set_b_evaluate_spawn_only(False)
            else:
                raise NameError("Orbit was not set to evaluate during spawn or update!")

            if options.get_editor_property("bUseEmitterTime"):
                convert_options.set_index_by_emitter_age()

            script_input = c2nUtils.create_script_input_for_distribution(distribution, convert_options)
            orbit_script.set_parameter(param_name, script_input)

        # set the offset.
        set_orbit_param(offset_amount_distribution, offset_options, "Offset")
        
        # set the rotation.
        set_orbit_param(rotation_distribution, rotation_options, "Rotation")
        
        # set the rotation rate.
        set_orbit_param(rotation_rate_distribution, rotation_rate_options, "Rotation Rate")
