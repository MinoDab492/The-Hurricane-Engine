'''
	This is an example of how to extend the Right Click menu in Rig Hierarchy window in Control Rig.

    This action, Add Null Above Selected, is accessible by selecting any number of controls, 
    use the right mouse click menu of the Rig Hierarchy -> New -> Add Null Above Selected

    In this example, we are adding the ability to create a null above each selected elements and
    maintain the offset parent transform.
'''

import unreal

# Each menu entry needs a custom menu owner
menu_owner = "ControlRigEditorExtension_RigHierarchy_AddNullAboveSelected"
tool_menus = unreal.ToolMenus.get()


def setup_menus():
    '''Creates the menu entry and adds to Right Click Context Menu on Rig Hierarchy'''
    menu = tool_menus.extend_menu("ControlRigEditor.RigHierarchy.ContextMenu.New")
    # Create a Python section within the menu
    menu.add_section("Python", "Python", insert_type=unreal.ToolMenuInsertType.AFTER)
    entry = add_null_above_selected()
    entry.init_entry(menu_owner,
                     "ControlRigEditor.RigHierarchy.ContextMenu.New",
                     "Python", "AddNullAbove", "Add Null Above Selected")
    menu.add_menu_entry_object(entry)


@unreal.uclass()
class add_null_above_selected(unreal.ToolMenuEntryScript):
    '''
    Custom Tool Menu Entry for RigNullElement creation

    Each override function will have a required context arg which will be used to find the
    specific ControlRigContextMenuContext, which will give the proper Control Rig in context.   
    '''

    @unreal.ufunction(override=True)
    def get_label(self, context):
        '''
        Override function for label of menu entry item'''
        return "Add Null Above Selected"

    @unreal.ufunction(override=True)
    def get_tool_tip(self, context):
        '''Override function for tool tip hover text'''
        return "Add a Null above selected controls, this will maintain the control's offset parent transform.\nPython Script found in: Engine/Plugins/Animation/ControlRig/Content/Python/RigHierarchy"

    @unreal.ufunction(override=True)
    def can_execute(self, context):
        '''Override function for if the menu entry can be executed'''
        control_rig_context = context.find_by_class(
            unreal.ControlRigContextMenuContext)
        if control_rig_context:
            control_rig_bp = control_rig_context.get_control_rig_blueprint()
            if control_rig_bp:
                selected_keys = control_rig_bp.hierarchy.get_selected_keys()

                if len(selected_keys) == 0:
                    return False

                for key in selected_keys:
                    if key.type != unreal.RigElementType.CONTROL:
                        return False

                return True
        return False

    @unreal.ufunction(override=True)
    def execute(self, context):
        '''Override function for the menu entry execution'''

        control_rig_context = context.find_by_class(
            unreal.ControlRigContextMenuContext)

        if control_rig_context:  # Confirm control rig context
            # Grab contextual control rig
            control_rig_bp = control_rig_context.get_control_rig_blueprint()
            if control_rig_bp:
                hierarchy = control_rig_bp.hierarchy
                hierarchy_controller = control_rig_bp.get_hierarchy_controller()
                selected_keys = hierarchy.get_selected_keys()

                for key in selected_keys:

                    transform_initial = hierarchy.get_global_transform(key, initial = True)
                    name = key.get_editor_property("name")

                    transform_name = str(name) + "_null" # Build the name of the null

                    # Query the parent, if none parent to world
                    parents = hierarchy.get_parents(key, recursive=False)
                    if parents:
                        parent = parents[0]
                    else:
                        parent = unreal.RigElementKey()

                    # Add a null at the transform
                    null = hierarchy_controller.add_null(name = transform_name, parent = parent, transform = transform_initial, transform_in_global=True, setup_undo= False, print_python_command=True)

                    # Reparent the control to the null
                    hierarchy_controller.set_parent(key, null, maintain_global_transform=True, setup_undo=True, print_python_command=True)
                    
                    # Set initial transform
                    hierarchy.set_global_transform(null, transform_initial, initial=True, affect_children=True, setup_undo=False, print_python_command=True)
                    hierarchy.set_global_transform(key, transform_initial, initial=True, affect_children=True, setup_undo=False, print_python_command=True)


def run():
    """
    Executes the creation of the menu entry
    Allow iterating on this menu python file without restarting editor
    """
    tool_menus.unregister_owner_by_name(menu_owner)
    setup_menus()
    tool_menus.refresh_all_widgets()
