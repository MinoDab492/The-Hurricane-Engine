from ModuleConverterInterface import ModuleConverterInterface
import unreal as ue
from unreal import FXConverterUtilitiesLibrary as ueFxUtils
import CascadeToNiagaraHelperMethods as c2nUtils
import Paths


class CascadeSizeMultiplyLifeConverter(ModuleConverterInterface):

    @classmethod
    def get_input_cascade_module(cls):
        return ue.ParticleModuleSizeMultiplyLife

    @classmethod
    def convert(cls, args):
        cascade_module = args.get_cascade_module()
        emitter = args.get_niagara_emitter_context()
        
        # choose the correct scale depending on the converted renderer
        renderer = emitter.find_renderer("RibbonRenderer")
        if renderer is not None:
            # noinspection PyTypeChecker
            cls.__convert_ribbon_size(cascade_module, emitter, renderer)
        elif emitter.find_renderer("MeshRenderer") is not None:
            # noinspection PyTypeChecker
            cls.__convert_mesh_size(cascade_module, emitter)
        elif emitter.find_renderer("SpriteRenderer") is not None:
            required_module = args.get_required_module()
            # noinspection PyTypeChecker
            cls.__convert_sprite_size(cascade_module, required_module, emitter)
        else:
            typedata_module = args.get_typedata_module()
            module_name = c2nUtils.get_module_name(cascade_module, "None")
            typedata_module_name = c2nUtils.get_module_name(typedata_module, "Sprite")
            raise NameError(
                "Could not convert module \"" + module_name + "\": CascadeSizeMultiplyLife converter does not support "
                "emitters with renderer of type \"" + typedata_module_name + "\".")
            
    @classmethod
    def __convert_ribbon_size(cls, cascade_module, emitter, renderer):
        """
        Convert size for ribbons.
        
        Args:
            cascade_module (ue.ParticleModuleSizeMultiplyLife)
            emitter (ue.NiagaraEmitterConversionContext)
            renderer (ue.NiagaraRendererProperties)
        """
        # get all properties from the cascade size multiply life module
        life_multiplier, multiply_x, multiply_y, multiply_z = \
            ueFxUtils.get_particle_module_size_multiply_life_props(cascade_module)
        
        if multiply_x:
            options = c2nUtils.DistributionConversionOptions()
            options.set_target_vector_component("x")
            life_input = c2nUtils.create_script_input_for_distribution(life_multiplier, options)
            
            script_asset = ueFxUtils.create_asset_data(Paths.di_multiply_float)
            script_args = ue.CreateScriptContextArgs(script_asset)
            scale_ribbon_script = ueFxUtils.create_script_context(script_args)
            
            ribbon_width_input = ueFxUtils.create_script_input_linked_parameter(
                "Particles.RibbonWidth",
                ue.NiagaraScriptInputType.FLOAT)
            scale_ribbon_script.set_parameter("A", ribbon_width_input)
            scale_ribbon_script.set_parameter("B", life_input)
            
            scale_ribbon_input = ueFxUtils.create_script_input_dynamic(scale_ribbon_script, ue.NiagaraScriptInputType.FLOAT)
            
            emitter.set_parameter_directly(
                "Particles.ScaledRibbonWidth",
                scale_ribbon_input,
                ue.ScriptExecutionCategory.PARTICLE_UPDATE)

            # rebind to the scaled ribbon width so there is not feedback from 
            # driving the value.
            emitter.set_renderer_binding(
                renderer,
                "RibbonWidthBinding",
                "Particles.ScaledRibbonWidth",
                ue.NiagaraRendererSourceDataMode.PARTICLES)

    @classmethod
    def __convert_mesh_size(cls, cascade_module, emitter):
        """
        Convert size for meshes

        Args:
            cascade_module (ue.ParticleModuleSizeMultiplyLife)
            emitter (ue.NiagaraEmitterConversionContext)
        """
        # get all properties from the cascade size multiply life module
        life_multiplier, multiply_x, multiply_y, multiply_z = \
            ueFxUtils.get_particle_module_size_multiply_life_props(cascade_module)

        # find/add the mesh scale script
        script_asset = ueFxUtils.create_asset_data(Paths.script_scale_mesh_size)
        script_args = ue.CreateScriptContextArgs(script_asset)
        scale_mesh_script = emitter.find_or_add_module_script(
            "ScaleMesh",
            script_args,
            ue.ScriptExecutionCategory.PARTICLE_UPDATE)
        
        # if multiplying through all components, do it normally
        if multiply_x and multiply_y and multiply_z:
            life_multiplier_input = c2nUtils.create_script_input_for_distribution(life_multiplier)
            scale_mesh_script.set_parameter("Scale Factor", life_multiplier_input)
            
        # get each component individually and combine
        else:
            emitter.log(
                "Scale mesh size by life module did not multiply all vector components: this mode is currently "
                "unsupported for conversion!",
                ue.NiagaraMessageSeverity.WARNING)
            
    @classmethod
    def __convert_sprite_size(cls, cascade_module, required_module, emitter):
        """
        Convert size for sprites.

        Args:
            cascade_module (ue.ParticleModuleSizeMultiplyLife)
            emitter (ue.NiagaraEmitterConversionContext)
        """
        # get all properties from the cascade size multiply life module
        (life_multiplier,
         multiply_x,
         multiply_y,
         multiply_z
         ) = ueFxUtils.get_particle_module_size_multiply_life_props(cascade_module)

        # find/add the sprite scale script
        script_args = ue.CreateScriptContextArgs(ueFxUtils.create_asset_data(Paths.script_scale_sprite_size), [1, 1])
        scale_sprite_script = emitter.find_or_add_module_script(
            "ScaleSprite",
            script_args,
            ue.ScriptExecutionCategory.PARTICLE_UPDATE)

        # if multiplying through all components, do it normally. Shrink from 
        # vec3 to vec2 as sprites do not have z dimension.
        if multiply_x and multiply_y:
            options = c2nUtils.DistributionConversionOptions()
            options.set_target_type_width(ue.NiagaraScriptInputType.VEC2)

            # if this is being used to drive a sprite renderer where the facing mode
            # is facing the camera, the only component of the particle size that is
            # needed is the x component.
            screen_alignment = required_module.get_editor_property("ScreenAlignment")

            if (
                screen_alignment == ue.ParticleScreenAlignment.PSA_SQUARE or
                screen_alignment == ue.ParticleScreenAlignment.PSA_FACING_CAMERA_POSITION or
                screen_alignment == ue.ParticleScreenAlignment.PSA_FACING_CAMERA_DISTANCE_BLEND
            ):
                options.set_target_vector_component("x")
            
            # if the y component is always 0 then expand the x component to 
            # scale uniformly across both axes.
            elif c2nUtils.distribution_always_equals(life_multiplier, 0.0, "y"):
                options.set_target_vector_component("x")
            
            life_multiplier_input = c2nUtils.create_script_input_for_distribution(life_multiplier, options)
            scale_sprite_script.set_parameter("Scale Factor", life_multiplier_input)

        # just the x or y component
        elif multiply_x or multiply_y:
            options = c2nUtils.DistributionConversionOptions()
            options.set_target_type_width(ue.NiagaraScriptInputType.FLOAT)
            if multiply_x:
                target_component = "x"
            else:  # multiply_y
                target_component = "y"
            options.set_target_vector_component(target_component)

            life_multiplier_input = c2nUtils.create_script_input_for_distribution(life_multiplier, options)
            script_asset = ueFxUtils.create_asset_data(Paths.di_break_vec2)
            script_args = ue.CreateScriptContextArgs(script_asset)
            break_vec2_script = ueFxUtils.create_script_context(script_args)
            break_vec2_script.set_parameter(target_component, life_multiplier_input)
            life_multiplier_input = ueFxUtils.create_script_input_dynamic(
                break_vec2_script,
                ue.NiagaraScriptInputType.VEC2)
            scale_sprite_script.set_parameter("Scale Factor", life_multiplier_input)
