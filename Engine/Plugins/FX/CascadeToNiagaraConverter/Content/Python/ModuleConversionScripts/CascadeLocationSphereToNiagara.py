from ModuleConverterInterface import ModuleConverterInterface
import unreal as ue
from unreal import FXConverterUtilitiesLibrary as ueFxUtils
import CascadeToNiagaraHelperMethods as c2nUtils
import Paths


class CascadeSphereLocationConverter(ModuleConverterInterface):

    @classmethod
    def get_input_cascade_module(cls):
        return ue.ParticleModuleLocationPrimitiveSphere

    @classmethod
    def convert(cls, args):
        cascade_module = args.get_cascade_module()
        emitter = args.get_niagara_emitter_context()

        # make the new module script for sphere location
        script_asset = ueFxUtils.create_asset_data(Paths.script_location_shape)
        script_args = ue.CreateScriptContextArgs(script_asset)
        sphere_location_script = emitter.find_or_add_module_script(
            "ShapeLocation",
            script_args,
            ue.ScriptExecutionCategory.PARTICLE_SPAWN)

        # get all properties from the cascade sphere location module
        # noinspection PyTypeChecker
        (radius,
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
         ) = ueFxUtils.get_particle_module_location_primitive_sphere_props(cascade_module)

        # make an input for radius
        radius_input = c2nUtils.create_script_input_for_distribution(radius)

        # Set shape to Sphere
        sphere_location_script.set_parameter(
            "Shape Primitive",
            ueFxUtils.create_script_input_enum(Paths.enum_shape_primitive, 'Sphere'))

        # copy over the radius
        sphere_location_script.set_parameter(
            "Sphere Radius",
            c2nUtils.create_script_input_for_distribution(radius))

        # set the sphere hemispheres if required.
        b_needs_scale = False
        hemisphere_distribution = ue.Vector2D(1.0, 1.0)
        yaw = 0
        scale = ue.Vector(1.0, 1.0, 1.0)

        # we handle the common cases of half circles and single quadrants, but ignore the rest as edge cases.
        # TODO: Should we handle all the cases?

        # Handle all the half circle cases
        if not b_positive_x and b_negative_x and b_positive_y and b_negative_y:
            hemisphere_distribution.x = 0.5
            yaw = 0.25
        elif b_positive_x and not b_negative_x and b_positive_y and b_negative_y:
            hemisphere_distribution.x = 0.5
            yaw = 0.75
        elif b_positive_x and b_negative_x and not b_positive_y and b_negative_y:
            hemisphere_distribution.x = 0.5
            yaw = 0.5
        elif b_positive_x and b_negative_x and b_positive_y and not b_negative_y:
            hemisphere_distribution.x = 0.5
            yaw = 0.0

        # Handle all the single quadrant cases
        elif b_positive_x and not b_negative_x and b_positive_y and not b_negative_y:
            hemisphere_distribution.x = 0.25
            yaw = 0.0
        elif not b_positive_x and b_negative_x and b_positive_y and not b_negative_y:
            hemisphere_distribution.x = 0.25
            yaw = 0.25
        elif not b_positive_x and b_negative_x and not b_positive_y and b_negative_y:
            hemisphere_distribution.x = 0.25
            yaw = 0.0
        elif b_positive_x and not b_negative_x and not b_positive_y and b_negative_y:
            hemisphere_distribution.x = 0.25
            yaw = 0.0

        # if we need to rotate to account for the hemispheres, apply it
        if not yaw == 0.0:
            sphere_location_script.set_parameter(
                "Rotation Mode",
                ueFxUtils.create_script_input_enum(Paths.enum_rotation_mode, 'Yaw / Pitch / Roll'))
            sphere_location_script.set_parameter(
                "Yaw / Pitch / Roll",
                ueFxUtils.create_script_input_vector(ue.Vector(yaw, 0.0, 0.0)),
                True,
                True)

        # We do handle the positive/negative z hemispheres here
        if b_positive_z and not b_negative_z:
            hemisphere_distribution.y = 0.5
        elif b_negative_z and not b_positive_z:
            hemisphere_distribution.y = 0.5
            b_needs_scale = True
            scale.z = -1.0
        elif not b_positive_z and not b_negative_z:
            b_needs_scale = True
            scale.z = 0.0

        # Set the new hemisphere distribution
        sphere_location_script.set_parameter(
            "Hemisphere Distribution",
            ueFxUtils.create_script_input_vec2(hemisphere_distribution),
            True,
            True)

        # if non uniform scale is required due to negative hemispheres, apply it.
        if b_needs_scale:
            sphere_location_script.set_parameter(
                "Non Uniform Scale",
                ueFxUtils.create_script_input_vector(scale),
                True,
                True)

        # set surface only emission if required.
        if b_surface_only:
            sphere_location_script.set_parameter(
                "Surface Only Band Thickness",
                ueFxUtils.create_script_input_float(0.0),
                True,
                True)

        # add velocity along the sphere if required.
        if b_velocity:
            # add a script to add velocity.
            script_asset = ueFxUtils.create_asset_data(Paths.script_add_velocity)
            script_args = ue.CreateScriptContextArgs(script_asset, [1, 2])
            add_velocity_script = emitter.find_or_add_module_script(
                "AddSphereVelocity",
                script_args,
                ue.ScriptExecutionCategory.PARTICLE_SPAWN)

            velocity_input = ueFxUtils.create_script_input_linked_parameter(
                "Particles.ShapeLocation.ShapeVector",
                ue.NiagaraScriptInputType.VEC3)

            # if there is velocity scaling, apply it.
            if c2nUtils.distribution_always_equals(velocity_scale_distribution, 0.0) is False:
                # make an input to calculate the velocity scale and index the scale by the emitter age.
                options = c2nUtils.DistributionConversionOptions()
                options.set_index_by_emitter_age()
                velocity_scale_input = c2nUtils.create_script_input_for_distribution(
                    velocity_scale_distribution,
                    options)

                # multiply the velocity by the scale.
                script_asset = ueFxUtils.create_asset_data(Paths.di_multiply_vector_by_float)
                script_args = ue.CreateScriptContextArgs(script_asset)
                multiply_vector_script = ueFxUtils.create_script_context(script_args)
                multiply_vector_script.set_parameter("Vector", velocity_input)
                multiply_vector_script.set_parameter("Float", velocity_scale_input)
                velocity_input = ueFxUtils.create_script_input_dynamic(
                    multiply_vector_script,
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
            start_location_input = c2nUtils.create_script_input_for_distribution(
                start_location_distribution,
                options)

            # set the start location.
            sphere_location_script.set_parameter("Offset", start_location_input)

            sphere_location_script.set_parameter(
                "Offset Mode",
                ueFxUtils.create_script_input_enum(Paths.enum_offset_mode, 'Default'))
