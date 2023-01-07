"""
This script will launch on Editor startup if Control Rig is enabled. The executions below are functional
examples on how users can use Python to extend the Control Rig Editor.
"""
import RigHierarchy.add_controls_for_selected
import RigHierarchy.add_null_above_selected
import RigHierarchy.align_items
import RigHierarchy.rename_items

RigHierarchy.add_controls_for_selected.run()
RigHierarchy.add_null_above_selected.run()
RigHierarchy.align_items.run()
RigHierarchy.rename_items.run()

if hasattr(unreal, 'RigVMUserWorkflowProvider'):
	import ControlRigWorkflows.workflow_aim_bone
	import ControlRigWorkflows.workflow_two_bone_ik
	ControlRigWorkflows.workflow_aim_bone.provider.register()
	ControlRigWorkflows.workflow_two_bone_ik.provider.register()