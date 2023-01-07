'''
	This is an example of how to extend the Right Click menu in Rig Hierarchy window in Control Rig.

    This action, Rename Items, is accessible by selecting any number of controls or nulls and using one
    of the entrys:
        search & replace
        add prefix
        add suffix
        rename

    In this example, its showing how users can extend the editor to rename rig hierarchy elements such as 
    controls and nulls.
'''

import unreal

# Each menu entry needs a custom menu owner
menu_owner = "ControlRigEditorExtension_RigHierarchy_Naming_Menu"
tool_menus = unreal.ToolMenus.get()

#==============================================================================
#   
#==============================================================================
def setup_menus():
    '''Creates the menu entry and adds to Right Click Context Menu on Rig Hierarchy'''
    
    menu = tool_menus.extend_menu("ControlRigEditor.RigHierarchy.ContextMenu")


    # Add a new menu
    naming_menu = menu.add_sub_menu(owner = "menu_owner", 
                                    section_name = "Elements",
                                    name = "Naming", 
                                    label = "Naming",
                                    tool_tip = "Naming toolset")

    # Add a 'Python' section to hold naming tools
    naming_menu.add_section("Python", "Python") 

    # Search and Replace Entry
    entry = search_replace_entry()
    entry.init_entry(menu_owner,
                    naming_menu.menu_name,
                    "Python", "Search_Replace_Names", "Search & Replace")
    naming_menu.add_menu_entry_object(entry)

    # Add Prefix Entry
    entry = add_prefix_entry()
    entry.init_entry(menu_owner,
                    naming_menu.menu_name,
                    "Python", "Add_Prefix", "Add Prefix")
    naming_menu.add_menu_entry_object(entry)

    # Add Suffix Entry
    entry = add_suffix_entry()
    entry.init_entry(menu_owner,
                    naming_menu.menu_name,
                    "Python", "Add_Suffix", "Add Suffix")
    naming_menu.add_menu_entry_object(entry)

    # Rename Entry
    entry = do_rename_entry()
    entry.init_entry(menu_owner,
                    naming_menu.menu_name,
                    "Python", "do_rename", "Rename Selected")
    naming_menu.add_menu_entry_object(entry)


#==============================================================================
#   Search & Replace Entry
#==============================================================================
@unreal.uclass()
class search_replace_name_dialog(unreal.Object):
    '''Custom class for RigControlElement creation settings'''
    replace = unreal.uproperty(str, meta={"DisplayName": "Replace"})
    search = unreal.uproperty(str, meta={"DisplayName": "Search"})

    def _post_init(self):
        '''Set default class values after initialization'''
        pass

@unreal.uclass()
class search_replace_entry(unreal.ToolMenuEntryScript):

    user_data = search_replace_name_dialog()

    @unreal.ufunction(override=True)
    def get_label(self, context):
        '''
        Override function for label of menu entry item'''
        return "Search & Replace"

    @unreal.ufunction(override=True)
    def get_tool_tip(self, context):
        '''Override function for tool tip hover text'''
        return "Search and replace text on multi-selected objects\nPython Script found in: Engine/Plugins/Animation/ControlRig/Content/Python/RigHierarchy"

    @unreal.ufunction(override=True)
    def can_execute(self, context):
        '''
        Override function for visbility of menu entry item

        This menu entry will only display if there are only bones in the selection
        '''
        control_rig_context = context.find_by_class(
            unreal.ControlRigContextMenuContext)
        if control_rig_context:
            control_rig_bp = control_rig_context.get_control_rig_blueprint()
            if control_rig_bp:
                selected_keys = control_rig_bp.hierarchy.get_selected_keys()

                if len(selected_keys) == 0:
                    return False

                for key in selected_keys:
                    if key.type == unreal.RigElementType.BONE:
                        return False

                return True
        return False

    @unreal.ufunction(override=True)
    def execute(self, context):

        ############################
        #  Get the context
        ############################
        control_rig_context = context.find_by_class(unreal.ControlRigContextMenuContext)
        object_details_view_options = unreal.EditorDialogLibraryObjectDetailsViewOptions()
        object_details_view_options.show_object_name = False
        object_details_view_options.allow_search = False
        confirmed = unreal.EditorDialog.show_object_details_view("Search & Replace Name",self.user_data,object_details_view_options)


        control_rig_bp = control_rig_context.get_control_rig_blueprint()
        if not control_rig_bp:
            raise Exception("Invalid Blueprint")

        hierarchy = control_rig_bp.hierarchy
        hierarchy_controller = control_rig_bp.get_hierarchy_controller()
        selected_keys = hierarchy.get_selected_keys() # Get selection
        selected_keys = hierarchy.sort_keys(selected_keys)  # Sorts by index

        ############################
        #  Do rename . . .
        ############################
        replaceStr = self.user_data.replace
        searchStr = self.user_data.search

        for each in selected_keys:
            each_type = each.get_editor_property("type")
            each_name = each.get_editor_property("name")

            # WE ONLY OPERATE ON CONTROLS AND NULLs for safety
            if each_type.name in ["BONE"]:
                unreal.log_warning("{} is a Bone. Only Controls and Nulls can be renamed. Please use F2 to rename bones.".format(each_name))
                continue

            new_name = str(each_name).replace(searchStr, replaceStr)

            # I got this from looking at the output log - awesome!
            hierarchy_controller.rename_element(unreal.RigElementKey(type=each_type, name = str(each_name)), new_name)

#==============================================================================
#   Add Prefix Entry
#==============================================================================
@unreal.uclass()
class add_prefix_dialog(unreal.Object):
    '''Custom class for RigControlElement creation settings'''
    prefix = unreal.uproperty(str, meta={"DisplayName": "New Prefix"})

@unreal.uclass()
class add_prefix_entry(unreal.ToolMenuEntryScript):

    user_data = add_prefix_dialog()

    @unreal.ufunction(override=True)
    def get_tool_tip(self, context):
        '''Override function for tool tip hover text'''
        return "Adds a prefix on multi-selected objects\nPython Script found in: Engine/Plugins/Animation/ControlRig/Content/Python/RigHierarchy"
    
    @unreal.ufunction(override=True)
    def can_execute(self, context):
        '''
        Override function for visbility of menu entry item

        This menu entry will only display if there are only bones in the selection
        '''
        control_rig_context = context.find_by_class(
            unreal.ControlRigContextMenuContext)
        if control_rig_context:
            control_rig_bp = control_rig_context.get_control_rig_blueprint()
            if control_rig_bp:
                selected_keys = control_rig_bp.hierarchy.get_selected_keys()

                if len(selected_keys) == 0:
                    return False

                for key in selected_keys:
                    if key.type == unreal.RigElementType.BONE:
                        return False

                return True
        return False

    @unreal.ufunction(override=True)
    def execute(self, context):
        ############################
        #  Get the context
        ############################
        control_rig_context = context.find_by_class(unreal.ControlRigContextMenuContext)
        object_details_view_options = unreal.EditorDialogLibraryObjectDetailsViewOptions()
        object_details_view_options.show_object_name = False
        object_details_view_options.allow_search = False

        confirmed = unreal.EditorDialog.show_object_details_view("Rename Selected",self.user_data,object_details_view_options)

        control_rig_bp = control_rig_context.get_control_rig_blueprint()

        if not control_rig_bp:
            raise Exception("Invalid Blueprint")

        if not confirmed:
            raise Exception("Invalid Dialog Input")

        hierarchy = control_rig_bp.hierarchy
        hierarchy_controller = control_rig_bp.get_hierarchy_controller()
        selected_keys = hierarchy.get_selected_keys() # Get selection
        selected_keys = hierarchy.sort_keys(selected_keys)  # Sorts by index
        
        ############################
        #  Do rename . . .
        ############################
        prefix = self.user_data.prefix

        for x, each in enumerate(selected_keys):
            each_type = each.get_editor_property("type")
            each_name = each.get_editor_property("name")

            # WE ONLY OPERATE ON CONTROLS AND NULLs for safety
            if each_type.name in ["BONE"]:
                unreal.log_warning("{} is a Bone. Only Controls and Nulls can be renamed. Please use F2 to rename bones.".format(each_name))
                continue
            
            # Setup the new name
            newString = prefix + str(each_name)

            # Rename the element
            hierarchy_controller.rename_element(unreal.RigElementKey(type=each_type, name = str(each_name)), newString)

#==============================================================================
#   Add Suffix Entry
#==============================================================================
@unreal.uclass()
class add_suffix_dialog(unreal.Object):
    '''Custom class for RigControlElement creation settings'''
    suffix = unreal.uproperty(str, meta={"DisplayName": "New Suffix"})

@unreal.uclass()
class add_suffix_entry(unreal.ToolMenuEntryScript):

    user_data = add_suffix_dialog()

    @unreal.ufunction(override=True)
    def get_tool_tip(self, context):
        '''Override function for tool tip hover text'''
        return "Adds a suffix on multi-selected objects\nPython Script found in: Engine/Plugins/Animation/ControlRig/Content/Python/RigHierarchy"
    
    @unreal.ufunction(override=True)
    def can_execute(self, context):
        '''
        Override function for visbility of menu entry item

        This menu entry will only display if there are only bones in the selection
        '''
        control_rig_context = context.find_by_class(
            unreal.ControlRigContextMenuContext)
        if control_rig_context:
            control_rig_bp = control_rig_context.get_control_rig_blueprint()
            if control_rig_bp:
                selected_keys = control_rig_bp.hierarchy.get_selected_keys()

                if len(selected_keys) == 0:
                    return False

                for key in selected_keys:
                    if key.type == unreal.RigElementType.BONE:
                        return False

                return True
        return False

    @unreal.ufunction(override=True)
    def execute(self, context):
        ############################
        #  Get the context
        ############################
        control_rig_context = context.find_by_class(unreal.ControlRigContextMenuContext)
        object_details_view_options = unreal.EditorDialogLibraryObjectDetailsViewOptions()
        object_details_view_options.show_object_name = False
        object_details_view_options.allow_search = False

        confirmed = unreal.EditorDialog.show_object_details_view("Rename Selected",self.user_data,object_details_view_options)

        control_rig_bp = control_rig_context.get_control_rig_blueprint()

        if not control_rig_bp:
            raise Exception("Invalid Blueprint")

        if not confirmed:
            raise Exception("Invalid Dialog Input")

        hierarchy = control_rig_bp.hierarchy
        hierarchy_controller = control_rig_bp.get_hierarchy_controller()
        selected_keys = hierarchy.get_selected_keys() # Get selection
        selected_keys = hierarchy.sort_keys(selected_keys)  # Sorts by index
        
        ############################
        #  Do rename . . .
        ############################
        suffix = self.user_data.suffix

        for x, each in enumerate(selected_keys):
            each_type = each.get_editor_property("type")
            each_name = each.get_editor_property("name")

            # WE ONLY OPERATE ON CONTROLS AND NULLs for safety
            if each_type.name in ["BONE"]:
                unreal.log_warning("{} is a Bone. Only Controls and Nulls can be renamed. Please use F2 to rename bones.".format(each_name))
                continue
            
            # Setup the new name
            newString = str(each_name) + suffix
            
            # Rename the element
            hierarchy_controller.rename_element(unreal.RigElementKey(type=each_type, name = str(each_name)), newString)

#==============================================================================
#   Add Rename Entry
#==============================================================================
@unreal.uclass()
class do_rename_dialog(unreal.Object):
    '''Custom class for RigControlElement creation settings'''
    newName = unreal.uproperty(str, meta={"DisplayName": "Name", "Tooltip": "Use '#' to add increment numbers fill.\nEG: spine_##_ctrl would be spine_01_ctrl, spine_02_ctrl, etc."})
    start_num = unreal.uproperty(int, meta={"DisplayName": "Start #"})
    
    
    def _post_init(self):
        '''Set default class values after initialization'''
        self.start_num = 1

@unreal.uclass()
class do_rename_entry(unreal.ToolMenuEntryScript):

    user_data = do_rename_dialog()

    @unreal.ufunction(override=True)
    def get_tool_tip(self, context):
        '''Override function for tool tip hover text'''
        return "Rename Selected: Use # to increment numbers on multi-selected objects. EG: spine_##_ctrl would be spine_01_ctrl, spine_02_ctrl, etc.\nPython Script found in: Engine/Plugins/Animation/ControlRig/Content/Python/RigHierarchy"

    @unreal.ufunction(override=True)
    def can_execute(self, context):
        '''
        Override function for visbility of menu entry item

        This menu entry will only display if there are only bones in the selection
        '''
        control_rig_context = context.find_by_class(
            unreal.ControlRigContextMenuContext)
        if control_rig_context:
            control_rig_bp = control_rig_context.get_control_rig_blueprint()
            if control_rig_bp:
                selected_keys = control_rig_bp.hierarchy.get_selected_keys()

                if len(selected_keys) == 0:
                    return False

                for key in selected_keys:
                    if key.type == unreal.RigElementType.BONE:
                        return False

                return True
        return False

    @unreal.ufunction(override=True)
    def execute(self, context):
        '''Override function for the menu entry execution'''

        # Display a message to users for usage in the output log
        unreal.log("Rename Selected: Use # to increment numbers on multi-selected objects.\nEG: spine_##_ctrl would be spine_01_ctrl, spine_02_ctrl, etc.")
    
        ############################
        #  Get the context
        ############################
        control_rig_context = context.find_by_class(unreal.ControlRigContextMenuContext)
        object_details_view_options = unreal.EditorDialogLibraryObjectDetailsViewOptions()
        object_details_view_options.show_object_name = False
        object_details_view_options.allow_search = False

        confirmed = unreal.EditorDialog.show_object_details_view("Rename Selected",self.user_data,object_details_view_options)

        control_rig_bp = control_rig_context.get_control_rig_blueprint()

        if not control_rig_bp:
            raise Exception("Invalid Blueprint")

        if not confirmed:
            raise Exception("Invalid Dialog Input")

        hierarchy = control_rig_bp.hierarchy
        hierarchy_controller = control_rig_bp.get_hierarchy_controller()
        selected_keys = hierarchy.get_selected_keys() # Get selection
        selected_keys = hierarchy.sort_keys(selected_keys)  # Sorts by index
        
        ############################
        #  Do rename . . .
        ############################
        newName = self.user_data.newName
        start_num = self.user_data.start_num
        origNum = currentNum = start_num
        
        ############################
        #  Find module
        ############################
        def find(s, ch):
            return [i for i, ltr in enumerate(s) if ltr == ch]

        ############################
        #  Rename each selected element. 
        #
        #  If a # is used it will be 
        #  replaced with a number.
        #       eg: my_control_##_l == my_control_01_l, my_control_02_l, etc.
        #
        ############################

        currentNum = 0

        for x, each in enumerate(selected_keys):
            each_type = each.get_editor_property("type")
            each_name = each.get_editor_property("name")

           
            if each_type.name in ["BONE"]:
                unreal.log_warning("{} is a Bone. Only Controls and Nulls can be renamed. Please use F2 to rename bones.".format(each_name))
                continue

            hashChars = find(newName, "#")
            newString = ''
            
            if not hashChars:
                # No hashes found, rename only...
                newString = newName
            else:
                # Found some hashes.
                # Replace with an incremental int
                for i, char in enumerate(newName):
                    if i == hashChars[-1]:
                        newString += str((origNum + currentNum) )
                    elif i in hashChars:
                        newString+='0'
                    else:
                        newString+=char
            currentNum += 1
            
            # Rename the element
            hierarchy_controller.rename_element(unreal.RigElementKey(type=each_type, name = str(each_name)), newString)


#==============================================================================
#   
#==============================================================================

def run():
    """
    Executes the creation of the menu entry
    Allow iterating on this menu python file without restarting editor
    """
    tool_menus.unregister_owner_by_name(menu_owner)
    setup_menus()
    tool_menus.refresh_all_widgets()