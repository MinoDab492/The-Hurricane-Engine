"""
Interface class for cascade emitter and system to niagara emitter and system properties converters.
"""
import unreal as ue


class EmitterPropertiesConverterInterface(object):

    @classmethod
    def get_name(cls):
        """
        Get a friendly name for the emitter properties converter to log.

        Returns:
            str: The friendly name string.
        """
        pass

    @classmethod
    def convert(
        cls,
        cascade_emitter,
        cascade_required_module,
        niagara_emitter_context
    ):
        """
        Convert the input cascade_emitter to niagara emitter properties.
        
        Args:
            cascade_emitter (ue.UParticleEmitter): 
            cascade_required_module (ue.UParticleModuleRequired): 
            niagara_emitter_context (ue.NiagaraEmitterConversionContext): The niagara emitter to be modified.
        """
        pass
