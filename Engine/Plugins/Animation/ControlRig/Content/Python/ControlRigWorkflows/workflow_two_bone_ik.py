'''
	This is an example of how to implement a workflow for a node in Python

    The action will set up the node's pin default values based on the selection
'''

import unreal

class provider:

    provider_handles = {}

    def __call__(self, in_node):
        # create a new workflow
        workflow = unreal.RigVMUserWorkflow()
        workflow.type = unreal.RigVMUserWorkflowType.NODE_CONTEXT
        workflow.title = 'Configure from Selection'
        workflow.tooltip = 'Configure the primary and secondary targets based on the selected elements'
        workflow.on_perform_workflow.bind_callable(self.perform_user_workflow)

        # choose the default options class. you can define your own classes
        # if you want to provide options to the user to choose from.
        workflow.options_class = unreal.ControlRigWorkflowOptions.static_class()

        # return a list of workflows for this provider
        return [workflow]

    def perform_user_workflow(self, in_options, in_controller):

        if not len(in_options.selection) in range(3, 6):
           in_options.report_error('Please select bone A, bone B, the effector (and optionally effector control and the pole vector control)!')
           return False

        node = in_options.subject

        # extract all of the elements from the selection
        bone_a_key = in_options.selection[0]
        bone_b_key = in_options.selection[1]
        effector_bone_key = in_options.selection[2] if len(in_options.selection) == 5 else unreal.RigElementKey()
        effector_ctrl_key = in_options.selection[3] if len(in_options.selection) == 5 else in_options.selection[2]
        pole_ctrl_key = (in_options.selection[4] if len(in_options.selection) == 5 else in_options.selection[3]) if len(in_options.selection) > 3 else unreal.RigElementKey()

        # get the transforms
        bone_a = in_options.hierarchy.get_global_transform(bone_a_key)
        bone_b = in_options.hierarchy.get_global_transform(bone_b_key)
        effector = in_options.hierarchy.get_global_transform(effector_ctrl_key)
        primary_axis = bone_b.make_relative(bone_a).translation.normal()

        # create a new node to base our settings on
        new_defaults = unreal.RigUnit_TwoBoneIKSimplePerItem()

        # init the settings based on the current defaults on the node
        # this makes sure we keep settings that we don't want to change
        new_defaults.import_text(node.get_struct_default_value())

        # setup the items on the node
        new_defaults.item_a = bone_a_key
        new_defaults.item_b = bone_b_key
        if effector_bone_key.name is None or effector_ctrl_key.type == unreal.RigElementType.BONE:
            new_defaults.effector_item = effector_ctrl_key if effector_bone_key.name is None else effector_bone_key
        else:
            new_defaults.effector_item = unreal.RigElementKey(unreal.RigElementType.BONE)

        # set the effector transform and bone lengths
        new_defaults.effector = effector
        new_defaults.primary_axis = primary_axis
        new_defaults.item_a_length = bone_b.make_relative(bone_a).translation.length()
        new_defaults.item_b_length = effector.make_relative(bone_b).translation.length()

        # configute the pole vector either to the default or to the provided ctrl
        if pole_ctrl_key.name is None:
            new_defaults.secondary_axis_weight = 0
            new_defaults.pole_vector = (bone_b.translation * 3 - bone_a.translation - effector.translation).normal()
            new_defaults.pole_vector_kind = unreal.ControlRigVectorKind.LOCATION
            new_defaults.pole_vector_space = unreal.RigElementKey(unreal.RigElementType.BONE)
        else:
            pole_vector = in_options.hierarchy.get_global_transform(pole_ctrl_key)
            secondary_axis = pole_vector.make_relative(bone_b).translation
            secondary_axis = secondary_axis - primary_axis * primary_axis.dot(secondary_axis)
            new_defaults.secondary_axis_weight = 1
            new_defaults.secondary_axis = secondary_axis
            new_defaults.pole_vector = unreal.Vector(0)
            new_defaults.pole_vector_kind = unreal.ControlRigVectorKind.LOCATION
            new_defaults.pole_vector_space = pole_ctrl_key

        # set the defaults on the new node based on our local unit struct
        if not in_controller.set_unit_node_defaults(node, new_defaults.export_text()):
            return False

        # hook up the effector control if needed
        if not effector_ctrl_key.name is None:

            # define a new get transform node
            get_transform_defaults = unreal.RigUnit_GetTransform()
            get_transform_defaults.item = effector_ctrl_key
            get_transform_defaults.space = unreal.BoneGetterSetterMode.GLOBAL_SPACE
            get_transform_defaults.initial = False

            # add the node
            get_transform_node = in_controller.add_unit_node_with_defaults(get_transform_defaults.static_struct(), get_transform_defaults.export_text(), 'Execute', node.get_position() + unreal.Vector2D(-340, 200))
            if get_transform_node is None:
                return False

            # link the get transform to the effector on the two bone ik node
            if not in_controller.add_link(get_transform_node.find_pin('Transform').get_pin_path(), node.find_pin('Effector').get_pin_path()):
                return False

        return True

    @classmethod
    def register(cls):
        # retrieve the node we want to add a workflow to
        unit_struct = unreal.RigUnit_TwoBoneIKSimplePerItem.static_struct()

        # create an empty callback and bind an instance of this class to it
        provider_callback = unreal.RigVMUserWorkflowProvider()
        provider_callback.bind_callable(cls())

        # remember the registered provider handle so we can unregister later
        handle = unreal.RigVMUserWorkflowRegistry.get().register_provider(unit_struct, provider_callback)

        # we also store the callback on the class 
        # so that it doesn't get garbage collected
        cls.provider_handles[handle] = provider_callback

    @classmethod
    def unregister(cls):
        for (handle, provider) in cls.provider_handles:
            unreal.RigVMUserWorkflowRegistry.get().unregister_provider(handle)
        cls.provider_handles = {}
