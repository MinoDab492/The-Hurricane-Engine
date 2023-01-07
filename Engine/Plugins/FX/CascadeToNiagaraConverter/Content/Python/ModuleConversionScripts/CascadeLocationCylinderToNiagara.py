from ModuleConverterInterface import ModuleConverterInterface
import unreal as ue
from unreal import FXConverterUtilitiesLibrary as ueFxUtils
import CascadeToNiagaraHelperMethods as c2nUtils
import Paths


class CascadeCylinderLocationConverter(ModuleConverterInterface):

    @classmethod
    def get_input_cascade_module(cls):
        return ue.ParticleModuleLocationPrimitiveCylinder

    @classmethod
    def convert(cls, args):
        cascade_module = args.get_cascade_module()
        emitter = args.get_niagara_emitter_context()

        # find/add the module script for cylinder location
        script_asset = ueFxUtils.create_asset_data(Paths.script_location_shape)
        script_args = ue.CreateScriptContextArgs(script_asset)
        cylinder_script = emitter.find_or_add_module_script(
            "ShapeLocation",
            script_args,
            ue.ScriptExecutionCategory.PARTICLE_SPAWN)

        # get all properties from the cascade cylinder location module
        # noinspection PyTypeChecker
        (b_radial_velocity,
         start_radius_distribution,
         start_height_distribution,
         height_axis,
         b_positive_x,
         b_positive_y,
         b_positive_z,
         b_negative_x,
         b_negative_y,
         b_negative_z,
         b_surface_only,
         b_velocity,
         velocity_scale_distribution,
         start_location_distribution
         ) = ueFxUtils.get_particle_module_location_primitive_cylinder_props(cascade_module)

        # Set shape to Cylinder
        cylinder_script.set_parameter(
            "Shape Primitive",
            ueFxUtils.create_script_input_enum(Paths.enum_shape_primitive, 'Cylinder'))

        # apply cylinder radius.
        # index the height and radius values by emitter age as this is evaluated
        # at spawn.
        options = c2nUtils.DistributionConversionOptions()
        options.set_index_by_emitter_age()
        radius_input = c2nUtils.create_script_input_for_distribution(start_radius_distribution, options)
        cylinder_script.set_parameter("Cylinder Radius", radius_input)

        # apply cylinder height.
        height_input = c2nUtils.create_script_input_for_distribution(start_height_distribution, options)
        cylinder_script.set_parameter("Cylinder Height", height_input)

        # set the orientation (height) axis.
        orientation_axis = ue.Vector(0.0, 0.0, 0.0)
        needs_orientation = False

        if height_axis == ue.CylinderHeightAxis.PMLPC_HEIGHTAXIS_X:
            orientation_axis.x = 1.0
            needs_orientation = True
        elif height_axis == ue.CylinderHeightAxis.PMLPC_HEIGHTAXIS_Y:
            orientation_axis.y = 1.0
            needs_orientation = True
        elif height_axis == ue.CylinderHeightAxis.PMLPC_HEIGHTAXIS_Z:
            orientation_axis.z = 1.0
            needs_orientation = True
        else:
            raise NameError("Failed to get valid height axis from cylinder "
                            "location module!")

        if needs_orientation:
            cylinder_script.set_parameter(
                "Rotation Mode",
                ueFxUtils.create_script_input_enum(Paths.enum_rotation_mode, 'Axis Angle'))
            cylinder_script.set_parameter(
                "Rotation Axis",
                ueFxUtils.create_script_input_vector(orientation_axis),
                True,
                True)

        # set the appropriate hemispheres depending on x,y,z bounds.
        b_needs_scale = False
        scale = ue.Vector(1.0, 1.0, 1.0)
        if b_positive_x and not b_negative_x:
            cylinder_script.set_parameter("Hemicircle X", ueFxUtils.create_script_input_bool(True))
        elif b_negative_x and not b_positive_x:
            cylinder_script.set_parameter("Hemicircle X", ueFxUtils.create_script_input_bool(True))
            b_needs_scale = True
            scale.x = -1.0

        if b_positive_y and not b_negative_y:
            cylinder_script.set_parameter("Hemicircle Y", ueFxUtils.create_script_input_bool(True))
        elif b_negative_y and not b_positive_y:
            cylinder_script.set_parameter("Hemicircle Y", ueFxUtils.create_script_input_bool(True))
            b_needs_scale = True
            scale.y = -1.0

        # note: positive/negative z are not factored into the cascade cylinder 
        # module.

        # if negative hemispheres were required, use the non uniform scale to 
        # flip the basis vector appropriately.
        if b_needs_scale:
            scale_input = ueFxUtils.create_script_input_vector(scale)
            cylinder_script.set_parameter(
                "Non Uniform Scale",
                scale_input,
                True,
                True)

        # set surface only emission if required.
        if b_surface_only:
            cylinder_script.set_parameter(
                "Surface Only Band Thickness",
                ueFxUtils.create_script_input_float(0.0),
                True,
                True)

        # add velocity along the cylinder if required.
        if b_velocity:
            # add a script to add velocity.
            script_asset = ueFxUtils.create_asset_data(Paths.script_add_velocity)
            script_args = ue.CreateScriptContextArgs(script_asset, [1, 2])
            add_velocity_script = emitter.find_or_add_module_script(
                "AddCylinderVelocity",
                script_args,
                ue.ScriptExecutionCategory.PARTICLE_SPAWN)

            velocity_input = ueFxUtils.create_script_input_linked_parameter(
                "Particles.ShapeLocation.ShapeVector",
                ue.NiagaraScriptInputType.VEC3)

            # if radial velocity is specified, zero the velocity component on 
            # the cylinder up vector.
            if b_radial_velocity:
                if height_axis == ue.CylinderHeightAxis.PMLPC_HEIGHTAXIS_X:
                    vector_mask_input = ueFxUtils.create_script_input_vector(ue.Vector(0.0, 1.0, 1.0))
                elif height_axis == ue.CylinderHeightAxis.PMLPC_HEIGHTAXIS_Y:
                    vector_mask_input = ueFxUtils.create_script_input_vector(ue.Vector(1.0, 0.0, 1.0))
                elif height_axis == ue.CylinderHeightAxis.PMLPC_HEIGHTAXIS_Z:
                    vector_mask_input = ueFxUtils.create_script_input_vector(ue.Vector(1.0, 1.0, 0.0))
                else:
                    raise NameError("Failed to get valid height axis from cylinder location module!")

                # mask the configured component of the velocity.
                script_asset = ueFxUtils.create_asset_data(Paths.di_multiply_vector)
                script_args = ue.CreateScriptContextArgs(script_asset)
                multiply_vector_script = ueFxUtils.create_script_context(script_args)
                multiply_vector_script.set_parameter("A", velocity_input)
                multiply_vector_script.set_parameter("B", vector_mask_input)
                velocity_input = ueFxUtils.create_script_input_dynamic(
                    multiply_vector_script,
                    ue.NiagaraScriptInputType.VEC3)

            # if there is velocity scaling, apply it.
            if c2nUtils.distribution_always_equals(velocity_scale_distribution, 0.0) is False:
                # make an input to calculate the velocity scale and index the 
                # scale by the emitter age.
                options = c2nUtils.DistributionConversionOptions()
                emitter_age_index = ueFxUtils.create_script_input_linked_parameter(
                    "Emitter.LoopedAge",
                    ue.NiagaraScriptInputType.FLOAT)
                options.set_custom_indexer(emitter_age_index)
                velocity_scale_input = c2nUtils.create_script_input_for_distribution(
                    velocity_scale_distribution,
                    options)

                # multiply the velocity by the scale.
                script_asset = ueFxUtils.create_asset_data(Paths.di_multiply_vector_by_float)
                script_args = ue.CreateScriptContextArgs(script_asset)
                mul_vec3_by_float_script = ueFxUtils.create_script_context(script_args)
                mul_vec3_by_float_script.set_parameter("Vector", velocity_input)
                mul_vec3_by_float_script.set_parameter("Float", velocity_scale_input)
                velocity_input = ueFxUtils.create_script_input_dynamic(
                    mul_vec3_by_float_script,
                    ue.NiagaraScriptInputType.VEC3)

            # apply the velocity.
            add_velocity_script.set_parameter("Velocity", velocity_input)

            # make sure we have a solve forces and velocity script.
            script_asset = ueFxUtils.create_asset_data(Paths.script_solve_forces_and_velocity)
            script_args = ue.CreateScriptContextArgs(script_asset)
            emitter.find_or_add_module_script(
                "SolveForcesAndVelocity",
                script_args,
                ue.ScriptExecutionCategory.PARTICLE_UPDATE)

        # offset the location if required.
        if c2nUtils.distribution_always_equals(start_location_distribution, 0.0) is False:
            # create an input to set the offset and index the value by emitter age.
            options = c2nUtils.DistributionConversionOptions()
            emitter_age_index = ueFxUtils.create_script_input_linked_parameter(
                "Emitter.LoopedAge",
                ue.NiagaraScriptInputType.FLOAT)
            options.set_custom_indexer(emitter_age_index)
            start_location_input = c2nUtils.create_script_input_for_distribution(start_location_distribution, options)

            # set the start location.
            cylinder_script.set_parameter("Offset", start_location_input)

            cylinder_script.set_parameter(
                "Offset Mode",
                ueFxUtils.create_script_input_enum(Paths.enum_offset_mode, 'Default'))
