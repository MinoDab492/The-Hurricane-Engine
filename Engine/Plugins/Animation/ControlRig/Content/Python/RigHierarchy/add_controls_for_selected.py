'''
	This is an example of how to extend the Right Click menu in Rig Hierarchy window in Control Rig.

    This action, Add Controls for Selected, is accessible by selecting any number of bones, 
    use the right mouse click menu of the Rig Hierarchy -> New -> Add Controls for Selected 

    In this example, we are adding the ability to create multiple controls from selected bones
    with different options. If the user has the alt key down while selecting the menu entry,
    an options window will appear.

    Options:
        prefix - an optional string prefix that will be added from selected items for the new controls
        suffix - an optional string suffix that will be added from selected items for the new controls, default is '_ctrl'
        output_format - an enum that determines what format the controls will be created in

    Output Format Enums:
        LIST - Created controls will have no parent and be in world
        HIERARCHY - Created controls will be created in a separate hierarchy stack from the selection
        CHILD - Created controls will be a child of each selected element
        
'''

import unreal

# Each menu entry needs a custom menu owner
menu_owner = "ControlRigEditorExtension_RigHierarchy_AddControlsForSelected"
tool_menus = unreal.ToolMenus.get()


def setup_menus():
    '''Creates the menu entry and adds to Right Click Context Menu on Rig Hierarchy'''
    menu = tool_menus.extend_menu("ControlRigEditor.RigHierarchy.ContextMenu.New")
    # Create a Python section within the menu
    menu.add_section("Python", "Python", insert_type=unreal.ToolMenuInsertType.AFTER)
    entry = add_controls_for_selected()
    entry.init_entry(menu_owner,
                     "ControlRigEditor.RigHierarchy.ContextMenu.New",
                     "Python", "AddControlsForSelected", "Add Controls For Selected")
    menu.add_menu_entry_object(entry)


@unreal.uenum()
class ControlOutputFormat(unreal.EnumBase):
    '''Custom Enum for RigControlElement creation format'''
    HIERARCHY = unreal.uvalue(0, meta={"DisplayName": "Hierarchy"})
    LIST = unreal.uvalue(1, meta={"DisplayName": "List"})
    CHILD = unreal.uvalue(2, meta={"DisplayName": "Child"})


@unreal.uclass()
class add_controls_for_selected_options(unreal.Object):
    '''Custom class for RigControlElement creation settings'''
    prefix = unreal.uproperty(str, meta={"DisplayName": "Prefix"})
    suffix = unreal.uproperty(str, meta={"DisplayName": "Suffix"})
    output_format = unreal.uproperty(ControlOutputFormat, meta={
                                     "DisplayName": "Output Format"})

    def _post_init(self):
        '''Set default class values after initialization'''
        self.output_format = ControlOutputFormat.HIERARCHY
        self.suffix = "_ctrl"
        self.invoke_settings_with_alt = False


@unreal.uclass()
class add_controls_for_selected(unreal.ToolMenuEntryScript):
    '''
    Custom Tool Menu Entry for RigControlElement creation

    Each override function will have a required context arg which will be used to find the
    specific ControlRigContextMenuContext, which will give the proper Control Rig in context.   
    '''
    user_data = add_controls_for_selected_options()

    @unreal.ufunction(override=True)
    def get_label(self, context):
        '''
        Override function for label of menu entry item

        With Alt button down, a different label will display
        '''
        control_rig_context = context.find_by_class(
            unreal.ControlRigContextMenuContext)
        if control_rig_context.is_alt_down():
            return "Add Controls For Selected Options"
        else:
            return "Add Controls For Selected"

    @unreal.ufunction(override=True)
    def get_tool_tip(self, context):
        '''Override function for tool tip hover text'''
        return "Add Controls from selected Bones, Alt+Click to access additional settings\nPython Script found in: Engine/Plugins/Animation/ControlRig/Content/Python/RigHierarchy"

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
                    if key.type != unreal.RigElementType.BONE:
                        return False

                return True
        return False

    @unreal.ufunction(override=True)
    def execute(self, context):
        '''Override function for the menu entry execution'''

        control_rig_context = context.find_by_class(
            unreal.ControlRigContextMenuContext)
        confirmed = False
        if control_rig_context.is_alt_down():
            # If alt is down, create a dialog with a details panel with
            # the add control for selected optionss
            object_details_view_options = unreal.EditorDialogLibraryObjectDetailsViewOptions()
            object_details_view_options.show_object_name = False
            object_details_view_options.allow_search = False
            confirmed = unreal.EditorDialog.show_object_details_view(
                "Add Controls For Selected Options",
                self.user_data,
                object_details_view_options)
        else:
            confirmed = True

        if confirmed:
            if control_rig_context:  # Confirm control rig context
                # Grab contextual control rig
                control_rig_bp = control_rig_context.get_control_rig_blueprint()
                if control_rig_bp:
                    hierarchy = control_rig_bp.hierarchy
                    hierarchy_controller = control_rig_bp.get_hierarchy_controller()
                    selected_keys = hierarchy.get_selected_keys()
                    selected_keys = hierarchy.sort_keys(
                        selected_keys)  # Sorts by index

                    default_setting = unreal.RigControlSettings()
                    default_setting.control_type = unreal.RigControlType.EULER_TRANSFORM
                    default_value = hierarchy.make_control_value_from_euler_transform(
                        unreal.EulerTransform(scale=[1, 1, 1]))

                    def to_string(name):
                        """Converts a name to string"""
                        return unreal.StringLibrary.conv_name_to_string(name)

                    def get_control_name(bone_key):
                        """Creates the control name from a given RigElementKey"""
                        return "{0}{1}{2}".format(
                            self.user_data.prefix,
                            to_string(bone_key.name),
                            self.user_data.suffix)

                    if self.user_data.output_format == ControlOutputFormat.LIST:
                        # Creates controls that be parented to world
                        for key in selected_keys:
                            control_name = get_control_name(key)
                            control_key = hierarchy_controller.add_control(control_name, unreal.RigElementKey(),
                                                                           default_setting, default_value, True, True)

                            transform = hierarchy.get_global_transform(
                                key, True)
                            hierarchy.set_control_offset_transform(
                                control_key, transform, True)

                    elif self.user_data.output_format == ControlOutputFormat.HIERARCHY:
                        # Creates controls that will be in a separate hierarchy stack from the selection
                        bone_to_control_map = {
                            unreal.Name(): unreal.RigElementKey()}
                        for index in range(len(selected_keys)):
                            key = selected_keys[index]
                            control_name = get_control_name(key)

                            selected_parent = unreal.RigElementKey()
                            parent = key

                            while True:
                                parents = hierarchy.get_parents(parent, False)
                                if len(parents) > 0:
                                    parent = parents[0]
                                    if parent in selected_keys:
                                        selected_parent = parent
                                        break  # Found an parent that is selected, exit
                                else:
                                    break  # No more parent to find, exit

                            # Create the transform values and calculate the offset from the parent
                            # then add to the newly created control
                            value = hierarchy.make_control_value_from_transform(
                                unreal.Transform())
                            control_key = hierarchy_controller.add_control(
                                control_name, bone_to_control_map[selected_parent.name], default_setting, default_value, True, True)

                            bone_to_control_map[key.name] = control_key

                            if selected_parent == unreal.RigElementKey():
                                parent_transform = unreal.Transform()
                            else:
                                parent_transform = hierarchy.get_global_transform(
                                    selected_parent, True)

                            child_transform = hierarchy.get_global_transform(
                                key, True)

                            offset_transform = unreal.MathLibrary.make_relative_transform(
                                child_transform, parent_transform)
                            hierarchy.set_control_offset_transform(
                                control_key, offset_transform, True)

                    elif self.user_data.output_format == ControlOutputFormat.CHILD:
                        # Creates controls that will be a child of each selected element
                        for key in selected_keys:
                            control_name = get_control_name(key)
                            hierarchy_controller.add_control(
                                control_name, key, default_setting, default_value, True, True)


def run():
    """
    Executes the creation of the menu entry
    Allow iterating on this menu python file without restarting editor
    """
    tool_menus.unregister_owner_by_name(menu_owner)
    setup_menus()
    tool_menus.refresh_all_widgets()
