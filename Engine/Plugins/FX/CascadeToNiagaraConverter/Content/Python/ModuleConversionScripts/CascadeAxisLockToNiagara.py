from ModuleConverterInterface import ModuleConverterInterface
import unreal as ue
from unreal import FXConverterUtilitiesLibrary as ueFxUtils


class CascadeAxisLockConverter(ModuleConverterInterface):

    @classmethod
    def get_input_cascade_module(cls):
        return ue.ParticleModuleOrientationAxisLock

    @classmethod
    def convert(cls, args):
        cascade_module = args.get_cascade_module()
        emitter = args.get_niagara_emitter_context()

        # lock axis is only valid for the sprite renderer, so start by finding 
        # that.
        sprite_renderer_props = emitter.find_renderer(
            "SpriteRenderer")
        if sprite_renderer_props is not None:
            # get all properties from the cascade orientation module
            # noinspection PyTypeChecker
            axis_lock = ueFxUtils.get_particle_module_orientation_axis_lock_props(cascade_module)
            
            # set the sprite facing vector and custom alignment vector depending
            # on the axis lock.
            if axis_lock == ue.ParticleAxisLock.EPAL_X:
                sprite_renderer_props.set_editor_property(
                    "FacingMode",
                    ue.NiagaraSpriteFacingMode.CUSTOM_FACING_VECTOR)
                emitter.set_parameter_directly(
                    "Particles.SpriteFacing",
                    ueFxUtils.create_script_input_vector(
                        ue.Vector(1.0, 0.0, 0.0)),
                    ue.ScriptExecutionCategory.PARTICLE_SPAWN)
                
            elif axis_lock == ue.ParticleAxisLock.EPAL_Y:
                sprite_renderer_props.set_editor_property(
                    "FacingMode",
                    ue.NiagaraSpriteFacingMode.CUSTOM_FACING_VECTOR)
                emitter.set_parameter_directly(
                    "Particles.SpriteFacing",
                    ueFxUtils.create_script_input_vector(
                        ue.Vector(0.0, 1.0, 0.0)),
                    ue.ScriptExecutionCategory.PARTICLE_SPAWN)
                
            elif axis_lock == ue.ParticleAxisLock.EPAL_Z:
                sprite_renderer_props.set_editor_property(
                    "FacingMode",
                    ue.NiagaraSpriteFacingMode.CUSTOM_FACING_VECTOR)
                emitter.set_parameter_directly(
                    "Particles.SpriteFacing",
                    ueFxUtils.create_script_input_vector(
                        ue.Vector(0.0, 0.0, 1.0)),
                    ue.ScriptExecutionCategory.PARTICLE_SPAWN)
                
            elif axis_lock == ue.ParticleAxisLock.EPAL_NEGATIVE_X:
                sprite_renderer_props.set_editor_property(
                    "FacingMode",
                    ue.NiagaraSpriteFacingMode.CUSTOM_FACING_VECTOR)
                emitter.set_parameter_directly(
                    "Particles.SpriteFacing",
                    ueFxUtils.create_script_input_vector(
                        ue.Vector(-1.0, 0.0, 0.0)),
                    ue.ScriptExecutionCategory.PARTICLE_SPAWN)
                
            elif axis_lock == ue.ParticleAxisLock.EPAL_NEGATIVE_Y:
                sprite_renderer_props.set_editor_property(
                    "FacingMode",
                    ue.NiagaraSpriteFacingMode.CUSTOM_FACING_VECTOR)
                emitter.set_parameter_directly(
                    "Particles.SpriteFacing",
                    ueFxUtils.create_script_input_vector(
                        ue.Vector(0.0, -1.0, 0.0)),
                    ue.ScriptExecutionCategory.PARTICLE_SPAWN)
                
            elif axis_lock == ue.ParticleAxisLock.EPAL_NEGATIVE_Z:
                sprite_renderer_props.set_editor_property(
                    "FacingMode",
                    ue.NiagaraSpriteFacingMode.CUSTOM_FACING_VECTOR)
                emitter.set_parameter_directly(
                    "Particles.SpriteFacing",
                    ueFxUtils.create_script_input_vector(
                        ue.Vector(0.0, 0.0, -1.0)),
                    ue.ScriptExecutionCategory.PARTICLE_SPAWN)
                
            elif axis_lock == ue.ParticleAxisLock.EPAL_ROTATE_X:
                sprite_renderer_props.set_editor_property(
                    "Alignment",
                    ue.NiagaraSpriteAlignment.CUSTOM_ALIGNMENT)
                emitter.set_parameter_directly(
                    "Particles.SpriteAlignment",
                    ueFxUtils.create_script_input_vector(
                        ue.Vector(1.0, 0.0, 0.0)),
                    ue.ScriptExecutionCategory.PARTICLE_SPAWN)
                
            elif axis_lock == ue.ParticleAxisLock.EPAL_ROTATE_Y:
                sprite_renderer_props.set_editor_property(
                    "Alignment",
                    ue.NiagaraSpriteAlignment.CUSTOM_ALIGNMENT)
                emitter.set_parameter_directly(
                    "Particles.SpriteAlignment",
                    ueFxUtils.create_script_input_vector(
                        ue.Vector(0.0, 1.0, 0.0)),
                    ue.ScriptExecutionCategory.PARTICLE_SPAWN)
                
            elif axis_lock == ue.ParticleAxisLock.EPAL_ROTATE_Z:
                sprite_renderer_props.set_editor_property(
                    "Alignment",
                    ue.NiagaraSpriteAlignment.CUSTOM_ALIGNMENT)
                emitter.set_parameter_directly(
                    "Particles.SpriteAlignment",
                    ueFxUtils.create_script_input_vector(
                        ue.Vector(0.0, 0.0, 1.0)),
                    ue.ScriptExecutionCategory.PARTICLE_SPAWN)
                
            else:
                raise NameError("Encountered invalid particle axis lock when "
                                "converting cascade lock axis module!")
