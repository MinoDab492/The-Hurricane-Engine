"""
Interface class for module converters.
"""
import unreal as ue


class ModuleConverterArgs:
    """
    Args struct for converter interface methods.
    """

    def __init__(
        self,
        cascade_module,
        required_module,
        typedata_module,
        niagara_emitter_context
    ):
        """
        Arguments for module converter method.

        Args:
            cascade_module (ue.ParticleModule): The main module to convert.
            required_module (ue.ParticleModuleRequired): The required module which all cascade emitters have.
            typedata_module (ue.ParticleModuleTypeDataBase): The typedata module which the cascade emitter may have.
            niagara_emitter_context (ue.NiagaraEmitterConversionContext): The niagara emitter to be modified.
        """
        self.__cascade_module = cascade_module
        self.__required_module = required_module
        self.__typedata_module = typedata_module
        self.__niagara_emitter_context = niagara_emitter_context

    def get_cascade_module(self) -> ue.ParticleModule:
        """
        Get the main module to convert.

        Returns:
            cascade_module (ue.ParticleModule): The main module to convert.
        """
        return self.__cascade_module

    def get_required_module(self) -> ue.ParticleModuleRequired:
        """
        Get the required module which all cascade emitters have.

        Returns:
            required_module (ue.ParticleModuleRequired): The required module which all cascade emitters have.
        """
        return self.__required_module

    def get_typedata_module(self) -> ue.ParticleModuleTypeDataBase:
        """
        Get the typedata module of the emitter being converted. 

        Returns:
            typedata_module (ue.ParticleModuleTypeDataBase): The typedata module of the emitter being converted.

        Notes: If the emitter being converted does not have a typedata module, this will return None.
        """
        return self.__typedata_module

    def get_niagara_emitter_context(self) -> ue.NiagaraEmitterConversionContext:
        """
        Get the niagara emitter to be modified.

        Returns:
            niagara_emitter_context (ue.NiagaraEmitterConversionContext): The niagara emitter to be modified.
        """
        return self.__niagara_emitter_context


class ModuleConverterInterface(object):
    """
    Abstract base class for module converters. Extend new ModuleConverters from this class and place them under the 
        CascadeToNiagaraConverter Plugin's Content/Python/ModuleConversionScripts directory to have it discovered.
    """

    @classmethod
    def get_input_cascade_module(cls) -> ue.Class:
        """
        Get the StaticClass() of the target derived Cascade UParticleModule to input.

        Returns:
            UClass: Return derived type of UParticleModule::StaticClass() to begin conversion from.
        """
        pass

    @classmethod
    def convert(cls, args):
        """
        Convert the cascade module to a niagara module.

        Args:
            args (ModuleConverterArgs): 
        """
        pass
