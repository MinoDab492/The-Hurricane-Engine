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

        if not len(in_options.selection) in range(2, 4):
           in_options.report_error('Please select aiming element, primary target (and optionally secondary target)!')
           return False

        node = in_options.subject

        # extract the transforms for the selection
        a = in_options.hierarchy.get_global_transform(in_options.selection[0])
        b = in_options.hierarchy.get_global_transform(in_options.selection[1])
        primary_local = b.make_relative(a)
        primary_axis = primary_local.translation.normal()

        # create a new node to base our settings on
        new_defaults = unreal.RigUnit_AimItem()

        # init the settings based on the current defaults on the node
        # this makes sure we keep settings that we don't want to change
        new_defaults.import_text(node.get_struct_default_value())

        # setup the basics - the item of the node as well as
        # resetting primary and secondary target
        new_defaults.item = in_options.selection[0]
        new_defaults.primary = unreal.RigUnit_AimItem_Target()
        new_defaults.secondary = unreal.RigUnit_AimItem_Target()

        # configure the primary target
        new_defaults.primary.weight = 1
        new_defaults.primary.axis = primary_axis
        new_defaults.primary.target = unreal.Vector(0)
        new_defaults.primary.space = in_options.selection[1]
        new_defaults.primary.kind = unreal.ControlRigVectorKind.LOCATION

        # configure the secondary target if we've selected enough elements
        if len(in_options.selection) == 2:
            new_defaults.secondary.weight = 0
        else:
            c = in_options.hierarchy.get_global_transform(in_options.selection[2])
            secondary_axis = c.make_relative(a).translation.normal()
            secondary_axis = -secondary_axis.cross(primary_axis).cross(primary_axis).normal()

            new_defaults.secondary.weight = 1
            new_defaults.secondary.axis = secondary_axis
            new_defaults.secondary.target = unreal.Vector(0)
            new_defaults.secondary.space = in_options.selection[2]
            new_defaults.secondary.kind = unreal.ControlRigVectorKind.LOCATION

        return in_controller.set_unit_node_defaults(node, new_defaults.export_text())

    @classmethod
    def register(cls):
        # retrieve the node we want to add a workflow to
        unit_struct = unreal.RigUnit_AimItem.static_struct()

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
