from ModuleConverterInterface import ModuleConverterInterface
import unreal as ue
from unreal import FXConverterUtilitiesLibrary as ueFxUtils
import CascadeToNiagaraHelperMethods as c2nUtils
import Paths


class CascadeLightConverter(ModuleConverterInterface):

    @classmethod
    def get_input_cascade_module(cls):
        return ue.ParticleModuleLight

    @classmethod
    def convert(cls, args):
        cascade_module = args.get_cascade_module()
        emitter = args.get_niagara_emitter_context()

        # get all properties from the cascade light module
        # noinspection PyTypeChecker
        (b_use_inverse_squared_falloff,
         b_affects_translucency,
         b_preview_light_radius,
         spawn_fraction,
         color_scale_over_life_distribution,
         brightness_over_life_distribution,
         radius_scale_distribution,
         light_exponent_distribution,
         lighting_channels,
         volumetric_scattering_intensity,
         b_high_quality_lights,
         b_shadow_casting_lights
         ) = ueFxUtils.get_particle_module_light_props(cascade_module)

        def convert_lq_light(emitter_):
            pass
            # add a light renderer to the emitter.
            light_renderer_props = ue.NiagaraLightRendererProperties()

            # set the inverse squared falloff property.
            light_renderer_props.set_editor_property("bUseInverseSquaredFalloff", b_use_inverse_squared_falloff)

            # if necessary, set the spawn fraction amount.
            spawn_message_verbose = True
            if spawn_fraction != 1.0:
                spawn_message_verbose = False
            emitter_.log(
                "Cascade light specified a spawn fraction but this mode is not supported by the asset converter.",
                ue.NiagaraMessageSeverity.WARNING,
                spawn_message_verbose)

            # if inverse squared falloff is not used, apply the light exponent.
            if (
                b_use_inverse_squared_falloff is False and
                c2nUtils.distribution_always_equals(light_exponent_distribution, 1.0) is False
            ):
                light_exponent_input = c2nUtils.create_script_input_for_distribution(light_exponent_distribution)
                emitter_.set_parameter_directly(
                    "Particles.LightExponent",
                    light_exponent_input,
                    ue.ScriptExecutionCategory.PARTICLE_UPDATE)

            # set whether to affect translucency.
            light_renderer_props.set_editor_property("bAffectsTranslucency", b_affects_translucency)

            # use conversion script to replicate Cascade's behavior with lights.
            script_asset = ueFxUtils.create_asset_data(Paths.script_light_properties)
            script_args = ue.CreateScriptContextArgs(script_asset)

            light_props_script = emitter.find_or_add_module_script(
                "Light Properties",
                script_args,
                ue.ScriptExecutionCategory.PARTICLE_UPDATE)

            # set the light color
            light_color_input = c2nUtils.create_script_input_for_distribution(color_scale_over_life_distribution)
            light_props_script.set_parameter("LightColorScale", light_color_input)

            # set the light brightness
            light_brightness_input = c2nUtils.create_script_input_for_distribution(brightness_over_life_distribution)
            light_props_script.set_parameter("LightBrightnessScale", light_brightness_input)

            # explicitly set the binding for light color as it defaults to
            # particles.color.
            emitter_.set_renderer_binding(
                light_renderer_props,
                "ColorBinding",
                "Particles.LightColor",
                ue.NiagaraRendererSourceDataMode.PARTICLES)

            # set the light radius.
            light_radius_input = c2nUtils.create_script_input_for_distribution(radius_scale_distribution)
            light_props_script.set_parameter("LightRadiusScale",  light_radius_input)

            # add the light renderer.
            emitter_.add_renderer("LightRenderer", light_renderer_props)

        def convert_hq_light(emitter_):
            # add a component renderer to the emitter.
            component_renderer_props = ue.NiagaraComponentRendererProperties()

            # set the component.
            component_renderer_props.set_editor_property("ComponentType", ue.PointLightComponent)

            # if necessary, set the spawn fraction amount.
            spawn_message_verbose = True
            if spawn_fraction != 1.0:
                spawn_message_verbose = False
            emitter_.log(
                "Cascade light specified a spawn fraction but this mode is not supported by the asset converter.",
                ue.NiagaraMessageSeverity.WARNING,
                spawn_message_verbose)

            # if inverse squared falloff is not used, apply the light exponent.
            if (
                b_use_inverse_squared_falloff is False and
                c2nUtils.distribution_always_equals(light_exponent_distribution, 1.0) is False
            ):
                light_exponent_input = c2nUtils.create_script_input_for_distribution(light_exponent_distribution)
                emitter_.set_parameter_directly(
                    "Particles.LightExponent",
                    light_exponent_input,
                    ue.ScriptExecutionCategory.PARTICLE_UPDATE)

            # set whether to affect translucency.
            component_renderer_props.set_editor_property( "bAffectsTranslucency", b_affects_translucency)

            # set the light color
            light_color_input = c2nUtils.create_script_input_for_distribution(color_scale_over_life_distribution)
            emitter_.set_parameter_directly(
                "Particles.LightColor",
                light_color_input,
                ue.ScriptExecutionCategory.PARTICLE_UPDATE)

            # explicitly set the binding for light color as it defaults to
            # particles.color.
            emitter_.set_renderer_binding(
                component_renderer_props,
                "ColorBinding",
                "Particles.LightColor",
                ue.NiagaraRendererSourceDataMode.PARTICLES)

            # set the light radius.
            light_radius_input = c2nUtils.create_script_input_for_distribution(radius_scale_distribution)
            emitter_.set_parameter_directly(
                "Particles.LightRadius",
                light_radius_input,
                ue.ScriptExecutionCategory.PARTICLE_UPDATE)

            # set the light brightness.
            #  todo implement light brightness

            # set the light shadowcasting flag.
            #  todo implement light shadowcasting flag

            # set the lighting channels.
            #  todo implement lighting channels.

            # add the light component renderer.
            emitter_.add_renderer("LightComponentRenderer", component_renderer_props)

        if b_high_quality_lights:
            convert_hq_light(emitter)
        else:
            convert_lq_light(emitter)
