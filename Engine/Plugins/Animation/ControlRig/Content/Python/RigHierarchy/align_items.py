'''
    This is an example of how to extend the Drag and Drop menu in Rig Hierarchy window in Control Rig.

    This action, Align, is accessible by selecting a control and holding down Shift + Alt + Left Mouse Drag to another element.
    
    In this example, we are adding the ability to align selected elements via drag and drop. On release,
    there are different options for alignment.

    Options:
        Parent - parent selected control to element where mouse focus was released
        Align - align Translate/Rotate/Scale of selected control to element where mouse focus was released,
                TRS will be applied on the Offset Transform globally.

'''

import unreal

menu_owner = "ControlRigEditorExtension_RigHierarchy_AlignItems"
tool_menus = unreal.ToolMenus.get()


def setup_menus():
    '''
    Creates the menu entry and adds to Drag and Drop Context Menu on Rig Hierarchy

    Additionally, there are submenus that will be created for each option as well
    '''
    menu = tool_menus.extend_menu(
        "ControlRigEditor.RigHierarchy.DragDropMenu.Align")

    entry = align_translation_rotation()
    entry.init_entry(menu_owner, "ControlRigEditor.RigHierarchy.DragDropMenu.Align",
                     "", "TranslationRotation", "Translation & Rotation")
    entry.data.insert_position.name = "ALL"
    entry.data.insert_position.position = unreal.ToolMenuInsertType.AFTER
    menu.add_menu_entry_object(entry)

    translation_menu = menu.add_sub_menu(
        menu_owner, "", "Translation", "Translation")

    entry = align_translation_all()
    entry.init_entry(
        menu_owner, "ControlRigEditor.RigHierarchy.DragDropMenu.Align.Translation", "", "All", "All")
    translation_menu.add_menu_entry_object(entry)

    entry = align_translation_x()
    entry.init_entry(
        menu_owner, "ControlRigEditor.RigHierarchy.DragDropMenu.Align.Translation", "", "X", "X")
    translation_menu.add_menu_entry_object(entry)

    entry = align_translation_y()
    entry.init_entry(
        menu_owner, "ControlRigEditor.RigHierarchy.DragDropMenu.Align.Translation", "", "Y", "Y")
    translation_menu.add_menu_entry_object(entry)

    entry = align_translation_z()
    entry.init_entry(
        menu_owner, "ControlRigEditor.RigHierarchy.DragDropMenu.Align.Translation", "", "Z", "Z")
    translation_menu.add_menu_entry_object(entry)

    entry = align_rotation()
    entry.init_entry(
        menu_owner, "ControlRigEditor.RigHierarchy.DragDropMenu.Align", "", "Rotation", "Rotation")
    entry.data.insert_position.name = "Translation"
    entry.data.insert_position.position = unreal.ToolMenuInsertType.AFTER
    menu.add_menu_entry_object(entry)

    entry = align_scale()
    entry.init_entry(
        menu_owner, "ControlRigEditor.RigHierarchy.DragDropMenu.Align", "", "Scale", "Scale")
    entry.data.insert_position.name = "Rotation"
    entry.data.insert_position.position = unreal.ToolMenuInsertType.AFTER
    menu.add_menu_entry_object(entry)


@unreal.uclass()
class align_translation_rotation(unreal.ToolMenuEntryScript):
    '''
    Custom Tool Menu Entry for RigControlElement Translation/Rotation alignment

    Each override function will have a required context arg which will be used to find the
    specific ControlRigContextMenuContext, which will give the proper Control Rig in context.   
    '''

    @unreal.ufunction(override=True)
    def execute(self, context):
        '''Override function for the menu entry execution'''
        rig_context = context.find_by_class(
            unreal.ControlRigContextMenuContext)

        def filter_function(subject, target):
            subject.translation = target.translation
            subject.rotation = target.rotation
            return subject

        align_elements(rig_context, filter_function)

    @unreal.ufunction(override=True)
    def can_execute(self, context):
        '''Override function for if the menu entry can be executed'''
        return True


@unreal.uclass()
class align_translation_all(unreal.ToolMenuEntryScript):
    '''
    Custom Tool Menu Entry for RigControlElement Translation alignment

    Each override function will have a required context arg which will be used to find the
    specific ControlRigContextMenuContext, which will give the proper Control Rig in context.   
    '''

    @unreal.ufunction(override=True)
    def execute(self, context):
        '''Override function for the menu entry execution'''
        rig_context = context.find_by_class(
            unreal.ControlRigContextMenuContext)

        def filter_function(subject, target):
            subject.translation = target.translation
            return subject

        align_elements(rig_context, filter_function)

    @unreal.ufunction(override=True)
    def can_execute(self, context):
        '''Override function for if the menu entry can be executed'''
        return True


@unreal.uclass()
class align_translation_x(unreal.ToolMenuEntryScript):
    '''
    Custom Tool Menu Entry for RigControlElement Translation X alignment

    Each override function will have a required context arg which will be used to find the
    specific ControlRigContextMenuContext, which will give the proper Control Rig in context.   
    '''

    @unreal.ufunction(override=True)
    def execute(self, context):
        '''Override function for the menu entry execution'''
        rig_context = context.find_by_class(
            unreal.ControlRigContextMenuContext)

        def filter_function(subject, target):
            subject.translation.x = target.translation.x
            return subject

        align_elements(rig_context, filter_function)

    @unreal.ufunction(override=True)
    def can_execute(self, context):
        '''Override function for if the menu entry can be executed'''
        return True


@unreal.uclass()
class align_translation_y(unreal.ToolMenuEntryScript):
    '''
    Custom Tool Menu Entry for RigControlElement Translation Y alignment

    Each override function will have a required context arg which will be used to find the
    specific ControlRigContextMenuContext, which will give the proper Control Rig in context.   
    '''

    @unreal.ufunction(override=True)
    def execute(self, context):
        '''Override function for the menu entry execution'''
        rig_context = context.find_by_class(
            unreal.ControlRigContextMenuContext)

        def filter_function(subject, target):
            subject.translation.y = target.translation.y
            return subject

        align_elements(rig_context, filter_function)

    @unreal.ufunction(override=True)
    def can_execute(self, context):
        '''Override function for if the menu entry can be executed'''
        return True


@unreal.uclass()
class align_translation_z(unreal.ToolMenuEntryScript):
    '''
    Custom Tool Menu Entry for RigControlElement Translation Z alignment

    Each override function will have a required context arg which will be used to find the
    specific ControlRigContextMenuContext, which will give the proper Control Rig in context.   
    '''

    @unreal.ufunction(override=True)
    def execute(self, context):
        '''Override function for the menu entry execution'''
        rig_context = context.find_by_class(
            unreal.ControlRigContextMenuContext)

        def filter_function(subject, target):
            subject.translation.z = target.translation.z
            return subject

        align_elements(rig_context, filter_function)

    @unreal.ufunction(override=True)
    def can_execute(self, context):
        '''Override function for if the menu entry can be executed'''
        return True


@unreal.uclass()
class align_rotation(unreal.ToolMenuEntryScript):
    '''
    Custom Tool Menu Entry for RigControlElement Rotation alignment

    Each override function will have a required context arg which will be used to find the
    specific ControlRigContextMenuContext, which will give the proper Control Rig in context.   
    '''

    @unreal.ufunction(override=True)
    def execute(self, context):
        '''Override function for the menu entry execution'''
        rig_context = context.find_by_class(
            unreal.ControlRigContextMenuContext)

        def filter_function(subject, target):
            subject.rotation = target.rotation
            return subject

        align_elements(rig_context, filter_function)

    @unreal.ufunction(override=True)
    def can_execute(self, context):
        '''Override function for if the menu entry can be executed'''
        return True


@unreal.uclass()
class align_scale(unreal.ToolMenuEntryScript):
    '''
    Custom Tool Menu Entry for RigControlElement Scale alignment

    Each override function will have a required context arg which will be used to find the
    specific ControlRigContextMenuContext, which will give the proper Control Rig in context.   
    '''

    @unreal.ufunction(override=True)
    def execute(self, context):
        '''Override function for the menu entry execution'''
        rig_context = context.find_by_class(
            unreal.ControlRigContextMenuContext)

        def filter_function(subject, target):
            subject.scale = target.scale
            return subject

        align_elements(rig_context, filter_function)

    @unreal.ufunction(override=True)
    def can_execute(self, context):
        '''Override function for if the menu entry can be executed'''
        return True


def align_elements(rig_context, filter_function):
    '''Sets the alignment based off of drag and drop context and filter '''

    if rig_context:
        # Use for changing the intial transforms
        rig_bp = rig_context.get_control_rig_blueprint()
        # Use for getting the current transforms and update the viewport
        rig = rig_context.get_control_rig()

        if rig_bp and rig:
            bp_hierarchy = rig_bp.hierarchy
            instance_hierarchy = rig.get_hierarchy()
            dragged_keys = rig_context.get_rig_hierarchy_drag_and_drop_context().dragged_element_keys
            target_key = rig_context.get_rig_hierarchy_drag_and_drop_context().target_element_key

            for dragged_key in dragged_keys:
                if dragged_key == target_key:  # Check if dragged element is the same as the dropped element
                    return

                if dragged_key.type == unreal.RigElementType.BONE:
                    bone_element = bp_hierarchy.find_bone(dragged_key)
                    if not bone_element.key.name.is_none():
                        parent_key = bp_hierarchy.get_first_parent(dragged_key)

                        # If this bone is a non-root, imported bone, show the warning
                        if bone_element.bone_type == unreal.RigBoneType.IMPORTED and parent_key.type == unreal.RigElementType.BONE:
                            warning_message = "Matching transforms of imported(white) bones can cause issues with animation"
                            warning_title = "Match Transform on Imported Bone"

                            # The setting name (the last argument) matches the name we used in SRigHierarchy.cpp for the same warning
                            result = unreal.EditorDialog.show_suppressable_warning_dialog(
                                warning_title, warning_message, "SRigHierarchy_Warning")
                            if result == False:
                                return

            # when drag and drop happens
            with unreal.ScopedEditorTransaction("Drag & Drop Align") as transaction:
                target_global_transform = instance_hierarchy.get_global_transform(
                    target_key, False)
                for dragged_key in dragged_keys:
                    new_dragged_key_transform = instance_hierarchy.get_global_transform(
                        dragged_key, False)

                    new_dragged_key_transform = filter_function(
                        new_dragged_key_transform, target_global_transform)  # grab transform from filter

                    if dragged_key.type == unreal.RigElementType.CONTROL:

                        # Make offset relative from the existing parent
                        parent_transform = instance_hierarchy.get_parent_transform(
                            dragged_key, False)
                        offset_transform = unreal.MathLibrary.make_relative_transform(
                            new_dragged_key_transform, parent_transform)

                        # Set the offset and zero out local
                        bp_hierarchy.set_control_offset_transform(
                            dragged_key, offset_transform, True, True, True, True)
                        bp_hierarchy.set_local_transform(
                            dragged_key, unreal.Transform(), True, True, True, True)

                    # All non controls can just be updated via their global transforms
                    else:
                        bp_hierarchy.set_global_transform(
                            dragged_key, new_dragged_key_transform, True, True, True, True)

                    # Update viewport control rig instance for visual feedback
                    instance_hierarchy.set_global_transform(
                        dragged_key, new_dragged_key_transform, False, True, True, True)


def run():
    """
    Executes the creation of the menu entry
    Allow iterating on this menu python file without restarting editor
    """
    tool_menus.unregister_owner_by_name(menu_owner)
    setup_menus()
    tool_menus.refresh_all_widgets()
