from ModuleConverterInterface import ModuleConverterInterface
import unreal as ue
from unreal import FXConverterUtilitiesLibrary as ueFxUtils
import CascadeToNiagaraHelperMethods as c2nUtils
import Paths


class CascadeParticleSizeConverter(ModuleConverterInterface):

    @classmethod
    def get_input_cascade_module(cls):
        return ue.ParticleModuleSize

    @classmethod
    def convert(cls, args):
        cascade_module = args.get_cascade_module()
        emitter_context = args.get_niagara_emitter_context()
        typedata_module = args.get_typedata_module()
        
        # choose the correct niagara module depending on the converted renderer
        if (
            typedata_module is None or  # sprite renderer
            type(typedata_module) == ue.ParticleModuleTypeDataGpu
        ):
            required_module = args.get_required_module()
            # noinspection PyTypeChecker
            cls.__convert_sprite_size(cascade_module, emitter_context, required_module)
        elif type(typedata_module) == ue.ParticleModuleTypeDataRibbon:
            # noinspection PyTypeChecker
            cls.__convert_ribbon_size(cascade_module, emitter_context)
        elif type(typedata_module) == ue.ParticleModuleTypeDataMesh:
            # noinspection PyTypeChecker
            cls.__convert_mesh_size(cascade_module, emitter_context)
        else:
            module_name = c2nUtils.get_module_name(cascade_module, "None")
            typedata_module_name = c2nUtils.get_module_name(typedata_module, "Sprite")
            raise NameError(
                "Could not convert module \"" + module_name + "\": CascadeParticleSizeConverter does not support "
                + "emitters with renderer of type \"" + typedata_module_name + "\".")
                
    @classmethod
    def __convert_sprite_size(
        cls,
        cascade_module,
        emitter_context,
        required_module
    ):
        """
        Convert particle size for sprites.
        
        Args:
            cascade_module (ue.ParticleModuleSize): 
            emitter_context (ue.NiagaraEmitterConversionContext): 
            required_module (ue.ParticleModuleRequired): 
        """
        # find/add the module script for init particle
       
        script_asset = ueFxUtils.create_asset_data(Paths.script_initialize_particle)
        script_version = [1, 0]
        script_args = ue.CreateScriptContextArgs(script_asset, script_version)
 
        initialize_particle_script = emitter_context.find_or_add_module_script(
            "InitializeParticle", 
            script_args,
            ue.ScriptExecutionCategory.PARTICLE_SPAWN)

        # get all properties from the cascade particle size module
        size_distribution = ueFxUtils.get_particle_module_size_props(cascade_module)

        # convert the size property
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

        size_input = c2nUtils.create_script_input_for_distribution(size_distribution, options)

        # set the size mode and size
        size_mode_input = ueFxUtils.create_script_input_enum(Paths.enum_niagara_size_scale_mode, "Non-Uniform")
        initialize_particle_script.set_parameter("Sprite Size Mode", size_mode_input)
        initialize_particle_script.set_parameter("Sprite Size", size_input, True, True)

    @classmethod
    def __convert_ribbon_size(cls, cascade_module, emitter_context):
        """
        Convert particle size for ribbons.

        Args:
            cascade_module (ue.ParticleModuleSize): 
            emitter_context (ue.NiagaraEmitterConversionContext): 
        """
        # find/add the module script for init ribbon
        script_args = ue.CreateScriptContextArgs(ueFxUtils.create_asset_data(Paths.script_initialize_ribbon), [1, 1])
        initialize_ribbon_script = emitter_context.find_or_add_module_script(
            "InitializeRibbon",
            script_args,
            ue.ScriptExecutionCategory.PARTICLE_SPAWN)
        
        # get all properties from the cascade particle size module
        size_distribution = ueFxUtils.get_particle_module_size_props(cascade_module)

        # convert the size property; break out the width component of the size vector
        options = c2nUtils.DistributionConversionOptions()
        options.set_target_type_width(ue.NiagaraScriptInputType.FLOAT)
        options.set_target_vector_component("x")     
        
        script_args = ue.CreateScriptContextArgs(ueFxUtils.create_asset_data(Paths.di_multiply_float), [1, 0])
        scale_ribbon_script = ueFxUtils.create_script_context(script_args)
        a_input = c2nUtils.create_script_input_for_distribution(size_distribution, options)
        scale_ribbon_script.set_parameter("A", a_input)
        scale_ribbon_script.set_parameter("B", ueFxUtils.create_script_input_float(2.0))
        size_input = ueFxUtils.create_script_input_dynamic(scale_ribbon_script, ue.NiagaraScriptInputType.FLOAT)
        
        # set the size
        initialize_ribbon_script.set_parameter("Ribbon Width", size_input)

    @classmethod
    def __convert_mesh_size(cls, cascade_module, emitter_context):
        """
        Convert particle size for meshes.

        Args:
            cascade_module (ue.ParticleModuleSize): 
            emitter_context (ue.NiagaraEmitterConversionContext): 
        """
        # find/add the module script for init particle
        script_asset = ueFxUtils.create_asset_data(Paths.script_initialize_particle)
        script_version = ue.NiagaraScriptVersion(1, 0)
        create_script_args = ue.CreateScriptContextArgs(script_asset, script_version)
        initialize_particle_script = emitter_context.find_or_add_module_script(
            "InitializeParticle",
            create_script_args,
            ue.ScriptExecutionCategory.PARTICLE_SPAWN)
    
        # get all properties from the cascade particle size module
        size_distribution = ueFxUtils.get_particle_module_size_props(cascade_module)
    
        # convert the size property
        size_input = c2nUtils.create_script_input_for_distribution(size_distribution)
    
        # set the size
        scale_mode_input = ueFxUtils.create_script_input_enum(Paths.enum_niagara_size_scale_mode, "Non-Uniform")
        initialize_particle_script.set_parameter("Mesh Scale Mode",scale_mode_input)
        initialize_particle_script.set_parameter("Mesh Scale", size_input)
