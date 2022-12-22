def class_has_parent(target_class, target_parent_class):
    """
    Check if a class has a specific parent class.

    Args:
        target_class (class): Class to check if it has a specific parent.
        target_parent_class (class): Class to compare to parent classes of
    target_class.

    Returns:
        bool: True if the target_class has target_parent_class as a parent.
    """

    parent_subclasses = set()
    classes_to_check = [target_parent_class]
    while classes_to_check:
        parent_class = classes_to_check.pop()
        for child_class in parent_class.__subclasses__():
            if child_class not in parent_subclasses:
                parent_subclasses.add(child_class)
                classes_to_check.append(child_class)
    return target_class in parent_subclasses
