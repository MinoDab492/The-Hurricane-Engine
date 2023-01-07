# Copyright Epic Games, Inc. All Rights Reserved.

""" Test (and demo) UE Python stub type hinting/syntax highlighting.

This module checks if Python stub type hinting works properly. The file can
be executed, but it is also meant to be loaded in a third party Python editor
that support type checking (VSCode/PyCharm) and syntax highlighting so that a
programmer can visually check if the hinted types and the syntax highlight is
working as expected.

WARNING: Ensure you select 'Type Checker' type hinting mode in the Editor Preferences
         to generate an accurate typing. The 'Auto-Completion' mode omits type
         coercion which will raise sevaral type check in thie file.

"""
import unreal
import typing
import random


def local_print(arg: typing.Any) -> None:
    """ Useful for debugging, but mostly to remove warnings about unused variables."""
    print(arg)


def take_sequence_types(seq_obj: typing.MutableSequence[typing.Any]) -> None:
    """ Test if unreal.Array and unreal.FixedArray can be passed to API expecting a typing.Sequence type. """
    local_print(seq_obj)


def take_iterable_types(iter_obj: typing.Iterable[typing.Any]) -> None:
    """ Test if containers can be passed to API expecting an iterable type. """
    local_print(iter_obj)


def take_set_types(set_obj: typing.MutableSet[typing.Any]) -> None:
    """ Test if unreal.Set can be passed to API expecting a typing.Set type. """
    local_print(set_obj)


def take_mapping_types(mapping_obj: typing.MutableMapping[typing.Any, typing.Any]) -> None:
    """ Test if unreal.Map can be passed to API expecting a typing.Mapping type. """
    local_print(mapping_obj)


class DelegateCallable:
    """ Used to verify if a callable method is properly added/called/removed from a multicast delegate. """
    def __init__(self) -> None:
        self.callback_count = 0

    def delegate_callback(self, int_param: int) -> int:
        self.callback_count += 1
        local_print(int_param)
        return self.callback_count

    def multicast_delegate_callback(self, str_param: str) -> None:
        self.callback_count += 1
        local_print(str_param)


def test_name_wrapper() -> None:
    from_default: unreal.Name = unreal.Name()
    local_print(from_default)

    from_another: unreal.Name = unreal.Name.cast(from_default)
    local_print(from_another)

    from_copy: unreal.Name = unreal.Name(from_default)
    local_print(from_copy)

    from_prop: object = unreal.PyTestStruct().get_editor_property("name")
    local_print(unreal.Name.cast(from_prop))

    from_str: unreal.Name = unreal.Name("Hi")
    valid: bool = from_str.is_valid()
    local_print(valid)

    from_str_cast: unreal.Name = unreal.Name.cast("None")
    none_value: bool = from_str_cast.is_none()
    local_print(none_value)


def test_text_wrapper() -> None:
    from_default: unreal.Text = unreal.Text()
    local_print(from_default)

    from_another: unreal.Text = unreal.Text.cast(from_default)
    local_print(from_another)

    from_copy: unreal.Text = unreal.Text(from_default)
    local_print(from_copy)

    from_prop: object = unreal.PyTestStruct().get_editor_property("text")
    local_print(unreal.Text.cast(from_prop))

    from_str: unreal.Text = unreal.Text("Hi")
    local_print(from_str)

    from_str_cast: unreal.Text = unreal.Text.cast("Some Text")
    local_print(from_str_cast)

    from_float: unreal.Text = unreal.Text.as_number(10.3)
    local_print(from_float)

    from_int: unreal.Text = unreal.Text.as_number(100)
    local_print(from_int)

    from_float_pct: unreal.Text = unreal.Text.as_percent(0.10)  # 10%
    local_print(from_float_pct)

    from_int_pct: unreal.Text = unreal.Text.as_percent(1)  # 100%
    local_print(from_int_pct)

    from_currency: unreal.Text = unreal.Text.as_currency(650, "US")  # $6.50
    local_print(from_currency)

    empty: bool = from_str.is_empty() or from_str.is_empty_or_whitespace()
    local_print(empty)

    transient: bool = from_str.is_transient()
    local_print(transient)

    culture_invariant: bool = from_str.is_culture_invariant()
    local_print(culture_invariant)

    from_string_table: bool = from_str.is_from_string_table()
    local_print(from_string_table)

    lower_str: unreal.Text = from_str.to_lower()
    local_print(lower_str)

    upper_str: unreal.Text = from_str.to_upper()
    local_print(upper_str)

    fmt_seq_args: unreal.Text = unreal.Text("Hello {0}! I'm {1} and I'm {2} years old")
    formatted_seq_args: unreal.Text = fmt_seq_args.format("Everybody", "Joe", 44)
    local_print(formatted_seq_args)

    fmt_mapping_args: unreal.Text = unreal.Text("Hello {a1}! I'm {a2} and I'm {a3} years old")
    formatted_mapping_args: unreal.Text = fmt_mapping_args.format({"a1": "Everybody", "a2": "Joe", "a3": 44})
    local_print(formatted_mapping_args)

    # NOTE: This version works, but the argument names are dynamic, so type checker issue a warning.
    fmt_named_args: unreal.Text = unreal.Text("Hello {a1}! I'm {a2} and I'm {a3} years old")
    formatted_named_args: unreal.Text = fmt_named_args.format(a1="Everybody", a2="Joe", a3=44)  # type: ignore
    local_print(formatted_named_args)


def test_delegate_wrapper() -> None:
    # Defines a callback that match FPyTestDelegate signature. (Need to check the C++ code for the signature)
    def callback(int_param: int) -> int:
        return int_param * 2

    from_default: unreal.PyTestDelegate = unreal.PyTestDelegate()
    local_print(from_default)

    from_another: unreal.PyTestDelegate = unreal.PyTestDelegate.cast(from_default)
    local_print(from_another)

    from_copy: unreal.PyTestDelegate = from_default.copy()
    local_print(from_copy)

    from_prop: object = unreal.PyTestObject().get_editor_property("delegate")
    local_print(unreal.PyTestDelegate.cast(from_prop))

    # is_bound(), bind_delegate(), bind_function(), bind_callable(), unbind(), execute() and execute_if_bound().
    o = unreal.PyTestObject()
    from_default.bind_function(o, "delegate_property_callback")  # PyTestObject has a UFUNCTION named DelegatePropertyCallback.
    from_default.bind_callable(callback)  # Replace the binding.
    from_copy.bind_delegate(from_default)
    if from_default.is_bound():
        value: int = from_default.execute(33)  # This delegate takes an int and returns an int.
        local_print(value)
        from_default.unbind()
    if from_copy.is_bound():
        value: int = from_copy.execute_if_bound(44)  # This delegate takes an int and returns an int.
        local_print(value)
        from_copy.unbind()


def test_multicast_delegate_wrapper() -> None:
    # Defines a callback that match FPyTestMulticastDelegate signature. (Need to check the C++ code for the signature)
    def callback(str_param: str) -> None:
        local_print(str_param)

    from_default: unreal.PyTestMulticastDelegate = unreal.PyTestMulticastDelegate()
    local_print(from_default)

    from_another: unreal.PyTestMulticastDelegate = unreal.PyTestMulticastDelegate.cast(from_default)
    local_print(from_another)

    from_copy: unreal.PyTestMulticastDelegate = from_default.copy()
    local_print(from_copy)

    from_prop: object = unreal.PyTestObject().get_editor_property("multicast_delegate")
    local_print(unreal.PyTestMulticastDelegate.cast(from_prop))

    bound: bool = from_default.is_bound()
    local_print(bound)

    o = unreal.PyTestObject()
    from_default.add_function(o, "multicast_delegate_property_callback")
    from_default.remove_function(o, "multicast_delegate_property_callback")
    from_default.add_function_unique(o, "multicast_delegate_property_callback")
    func_in: bool = from_default.contains_function(o, "multicast_delegate_property_callback")
    from_default.remove_object(o)
    from_default.add_callable(callback)
    from_default.remove_callable(callback)
    from_default.add_callable_unique(callback)
    callable_in: bool = from_default.contains_callable(callable)
    from_default.remove_callable(callback)
    from_default.clear()
    bound: bool = from_default.is_bound()
    if func_in or callable_in or bound:
        from_default.broadcast("hi")


def test_field_type_wrapper() -> None:
    from_default: unreal.FieldPath = unreal.FieldPath()
    local_print(from_default)

    from_another: unreal.FieldPath = unreal.FieldPath(from_default)
    local_print(from_another)

    from_copy: unreal.FieldPath = from_default.copy()
    local_print(from_copy)

    from_prop: object = unreal.PyTestObject().get_editor_property("field_path")
    local_print(unreal.FieldPath.cast(from_prop))

    from_str: unreal.FieldPath = unreal.FieldPath("some_path")
    local_print(from_str)

    from_str_cast: unreal.FieldPath = unreal.FieldPath.cast("some_path")
    local_print(from_str_cast)

    valid: bool = from_str.is_valid()
    local_print(valid)


def test_enum_wrapper() -> None:
    from_value: unreal.PyTestEnum = unreal.PyTestEnum.ONE
    local_print(from_value)

    from_another: unreal.PyTestEnum = unreal.PyTestEnum.cast(from_value)
    local_print(from_another)

    from_prop: object = unreal.PyTestObject().get_editor_property("enum")
    local_print(unreal.PyTestEnum.cast(from_prop))

    static_enum: unreal.Enum = unreal.PyTestEnum.static_enum()
    local_print(static_enum)

    name: unreal.Text = from_value.get_display_name()
    local_print(name)


def test_struct_wrapper() -> None:
    """ Ensures that the struct wrapper is correctly hinted. """
    from_default: unreal.PyTestStruct = unreal.PyTestStruct()
    local_print(from_default)

    from_another: unreal.PyTestStruct = unreal.PyTestStruct.cast(from_default)
    local_print(from_another)

    from_prop: object = unreal.PyTestObject().get_editor_property("struct")
    local_print(unreal.PyTestStruct.cast(from_prop))

    from_dict: unreal.PyTestStruct = unreal.PyTestStruct.cast({"int": 20, "string": "joe"})  # Partial mapping.
    local_print(from_dict)

    from_seq: unreal.PyTestStruct = unreal.PyTestStruct.cast([True, 20, 6.4, unreal.PyTestEnum.TWO, "joe"])  # Partial sequence.
    local_print(from_seq)

    from_tuple: unreal.PyTestStruct = unreal.PyTestStruct.cast(from_default.to_tuple())
    local_print(from_tuple)

    from_upcast: unreal.PyTestStruct = unreal.PyTestStruct.cast(unreal.PyTestChildStruct())
    local_print(from_upcast)

    from_downcast: unreal.PyTestChildStruct = unreal.PyTestChildStruct.cast(from_upcast)
    local_print(from_downcast)

    script_struct: unreal.ScriptStruct = unreal.PyTestStruct.static_struct()
    print(script_struct)

    # assign()
    s: unreal.PyTestStruct = unreal.PyTestStruct()
    assign_from_dict = s.assign({"string": "foo"})
    local_print(assign_from_dict)
    assign_from_seq = s.assign([True, 20, 6.4, unreal.PyTestEnum.TWO, "joe"])
    local_print(assign_from_seq)
    assign_from_other = s.assign(from_seq)
    local_print(assign_from_other)
    assign_from_derived = s.assign(unreal.PyTestChildStruct())
    local_print(assign_from_derived)

    values: typing.Tuple[object, ...] = s.to_tuple()
    local_print(values)

    prop: object = s.get_editor_property(name="string_array")
    arr_prop: unreal.Array[str] = unreal.Array.cast(str, prop)
    print(arr_prop)
    s.set_editor_property(name="text", value=unreal.Text("some text"), notify_mode=unreal.PropertyAccessChangeNotifyMode.DEFAULT)
    s.set_editor_properties({"int": 20, "string": "joe"})

    exported_text: str = s.export_text()
    s.import_text(content=exported_text)


def test_object_wrapper() -> None:
    """ Ensures that the object wrapper class is correctly hinted. """
    from_default: unreal.PyTestObject = unreal.PyTestObject()
    local_print(from_default)

    from_another: unreal.PyTestObject = unreal.PyTestObject.cast(from_default)
    local_print(from_another)

    from_prop: object = unreal.PyTestTypeHint().get_editor_property("object_prop")
    local_print(unreal.PyTestObject.cast(from_prop))

    from_new: unreal.PyTestObject = unreal.PyTestObject.cast(unreal.new_object(unreal.PyTestObject.static_class()))
    local_print(from_new)

    from_upcast: unreal.PyTestObject = unreal.PyTestObject.cast(unreal.new_object(unreal.PyTestChildObject.static_class()))  # Upcast
    local_print(from_upcast)

    from_downcast: unreal.PyTestChildObject = unreal.PyTestChildObject.cast(from_upcast)  # Downcast
    local_print(from_downcast)

    from_cdo: unreal.PyTestObject = unreal.PyTestObject.get_default_object()
    local_print(from_cdo)

    # static_class()/get_class()
    static_cls: unreal.Class = unreal.PyTestObject.static_class()
    local_print(static_cls)
    instance_cls: unreal.Class = from_default.get_class()
    local_print(instance_cls)

    # get_outer()/get_typed_outer()/get_outermost()
    outer: unreal.Object = from_default.get_outer()
    local_print(outer)
    typed_outer1: unreal.Object = from_default.get_typed_outer(unreal.Object)
    typed_outer2: unreal.Object = from_default.get_typed_outer(unreal.Object.static_class())
    local_print(typed_outer1)
    local_print(typed_outer2)
    outermost: unreal.Package = from_default.get_outermost()
    local_print(outermost)

    external_pkg: bool = from_default.is_package_external()
    local_print(external_pkg)

    pkg: unreal.Package = from_default.get_package()
    local_print(pkg)

    name: str = from_default.get_name()
    local_print(name)

    fname: unreal.Name = from_default.get_fname()
    local_print(fname)

    fullname: str = from_default.get_full_name()
    local_print(fullname)

    pathname: str = from_default.get_path_name()
    local_print(pathname)

    world: typing.Optional[unreal.World] = from_default.get_world()
    local_print(world)

    modified: bool = from_default.modify(True)
    local_print(modified)

    renamed: bool = unreal.PyTestObject().rename(name="NewName" + str(random.randint(0, 1000000)), outer=None)
    local_print(renamed)

    from_default.set_editor_property(name="String", value="foo", notify_mode=unreal.PropertyAccessChangeNotifyMode.DEFAULT)
    from_default.set_editor_properties({"String": "foo", "Int": 14})
    prop: object = from_default.get_editor_property(name="String")
    local_print(prop)

    retval: int = from_default.call_method("CallFuncBlueprintNative", (1,))
    local_print(retval)
    retval: int = from_default.call_method("CallFuncBlueprintNative", kwargs={"value": 1})
    local_print(retval)
    

def test_fixed_array_wrapper() -> None:
    a: unreal.FixedArray[float] = unreal.FixedArray(type=float, len=10)
    local_print(a)

    from_another: unreal.FixedArray[float] = unreal.FixedArray.cast(float, a)
    local_print(from_another)

    from_copy: unreal.FixedArray[float] = a.copy()
    local_print(from_copy)

    from_list: unreal.FixedArray[int] = unreal.FixedArray.cast(type=int, obj=[1, 2, 3])
    local_print(from_list)

    from_iterable: unreal.FixedArray[int] = unreal.FixedArray.cast(int, {0: "0", 1: "1"}.keys())
    local_print(from_iterable)

    # unreal.FixedArray can be passed as typing.Sequence/typing.Iterable type.
    take_sequence_types(a)
    take_iterable_types(a)

    # __setitem__(self, index: int, value: T) -> None:
    a[0] = 10.0

    # __getitem__(self, index: int) -> T
    flt: float = a[0]
    local_print(flt)

    # Check polymorphism.
    poly: unreal.FixedArray[unreal.Object] = unreal.FixedArray(unreal.Object, 2)
    poly[0] = unreal.Actor()


def test_array_wrapper() -> None:
    a: unreal.Array[str] = unreal.Array(type=str)

    from_another: unreal.Array[str] = unreal.Array.cast(type=str, obj=a)
    local_print(from_another)

    from_copy: unreal.Array[str] = a.copy()
    local_print(from_copy)

    from_prop: object = unreal.PyTestObject().get_editor_property("string_array")
    local_print(unreal.Array.cast(str, from_prop))

    from_list: unreal.Array[int] = unreal.Array.cast(int, [1.1, 2.2, 3.3])  # Converting floats into ints.
    local_print(from_list)

    from_tuple: unreal.Array[int] = unreal.Array.cast(int, (1, 2, 3)) 
    local_print(from_tuple)

    from_iterable: unreal.Array[int] = unreal.Array.cast(int, {0: "0", 1: "1"}.keys())
    local_print(from_iterable)

    # unreal.FixedArray can be passed as typing.Sequence/typing.Iterable type.
    take_sequence_types(a)
    take_iterable_types(a)

    # append(self, value: T) -> None:
    a.append("Hello") 

    # count(self, value: T) -> int:
    local_print(a.count("b") == 1)  # Print a bool

    # extend(self, iterable: Iterable[T]) -> None:
    a.extend(["a", "b"])
    a.extend({"0": 10, "1": 20, "3": 30}.keys())

    # index(self, value: T, start: int = 0, stop: int = -1) -> int
    index: int = a.index(value="b", start=0, stop=-1)
    local_print(index)

    # insert(self, index: int, value: T) -> None:
    a.insert(index=0, value="foo")

    # pop(self, index: int = -1) -> T:
    pop_value: str = a.pop(0)
    local_print(pop_value)

    # remove(self, value: T) -> None:
    a.remove("b")

    # reverse(self) -> None:
    a.reverse()

    # sort(self, key: Optional[Callable[T]] = None, reverse: bool = False) -> None:
    a.sort(key=lambda item: len(item), reverse=True)
    a.sort(key=None, reverse=True)

    # resize(self, len: int) -> None:
    a.resize(22)

    # __setitem__/__getitem__
    a[0] = "Bingo"
    value: str = a[0]
    local_print(value)

    name_array: unreal.Array[typing.Union[str, unreal.Name]] = unreal.Array(unreal.Name)
    name_array.append("hello")  # string are convertible to unreal.Name().

    text_array: unreal.Array[typing.Union[str, unreal.Text]] = unreal.Array(unreal.Text)
    text_array.append("hello")  # string are convertible to unreal.Text().

    object_array: unreal.Array[typing.Optional[unreal.Object]] = unreal.Array(unreal.Object)
    object_array.append(None)
    object_array.append(unreal.PyTestObject())  # Polymorphism.


def test_set_wrapper() -> None:
    s: unreal.Set[int] = unreal.Set(type=int)

    from_another = unreal.Set.cast(type=int, obj=s)
    local_print(from_another)

    from_copy = s.copy()
    local_print(from_copy)

    from_prop: object = unreal.PyTestObject().get_editor_property("string_set")
    local_print(unreal.Set.cast(str, from_prop))

    from_set: unreal.Set[int] = unreal.Set.cast(int, {1, 2, 3}) 
    local_print(from_set)

    from_list: unreal.Set[int] = unreal.Set.cast(int, [1.1, 2.2, 3.3])  # Converting floats into ints.
    local_print(from_list)

    from_tuple: unreal.Set[int] = unreal.Set.cast(int, (1, 2, 3)) 
    local_print(from_tuple)

    from_iterable: unreal.Set[int] = unreal.Set.cast(int, {0: "0", 1: "1"}.keys())
    local_print(from_iterable)

    # unreal.Set can be passed to API expecting a typing.Set type.
    take_set_types(s)
    take_iterable_types(s)

    s.add(0)
    s.add(1)
    s.discard(0)
    s.add(0)
    s.remove(0)
    poped_out: int = s.pop()
    local_print(poped_out)
    s.clear()

    s.difference_update({2, 3}, [3, 4])
    diff: unreal.Set[int] = s.difference({2, 3}, [3, 4])
    local_print(diff)

    s.intersection_update([2, 3, 4, 8, 9])
    intersection: unreal.Set[int] = s.intersection({1, 2}, [5, 6])  # Good
    local_print(intersection)

    s.symmetric_difference_update(from_iterable)
    symmetric_diff: unreal.Set[int] = s.symmetric_difference(from_iterable)
    local_print(symmetric_diff)

    s.update({2, 3}, [3, 4], (6, 12, 13))
    union: unreal.Set[int] = s.union({2, 3}, [3, 4], (6, 12, 13))
    local_print(union)

    disjoint: bool = s.isdisjoint(from_iterable)
    local_print(disjoint)

    subset: bool = s.issubset(from_iterable)
    local_print(subset)

    superset: bool = s.issubset(from_iterable)
    local_print(superset)

    # Check polymorphism.
    poly: unreal.Set[unreal.Object] = unreal.Set(unreal.Object)
    poly.add(unreal.Actor())


def test_map_wrapper() -> None:
    local_print("== test_map_hinting ==")

    # __init__(self, keys_type: type, values_type: type) -> None:
    m: unreal.Map[int, str] = unreal.Map(key=int, value=str)

    # unreal.Map can be passed to API expecting a typing.Mapping type.
    take_mapping_types(m)

    # __setitem__(self, key: KT, value: VT) -> None:
    m[0] = "a"
    m[1] = "b"

    # __getitem__(self, key: KT) -> VT:
    value: str = m.__getitem__(1)
    local_print(value)

    # cast(cls, keys_type: Type[KT], values_type: Type[VT], obj: Any) -> Map[KT, VT]:
    m2: unreal.Map[int, str] = unreal.Map.cast(key=int, value=str, obj={0: "A", 1: "B"})
    local_print(m2)
    prop: object = unreal.PyTestObject().get_editor_property("string_int_map")
    local_print(unreal.Map.cast(str, int, prop))

    # __copy__(self) -> Map[KT, VT]:
    m3: unreal.Map[int, str] = m2.copy()
    local_print(m3)

    # fromkeys(cls, iterable: Iterable[KT], value: Optional[VT] = None) -> Map[KT, VT]:
    m4: unreal.Map[str, float] = unreal.Map.fromkeys(sequence=["A", "B", "C"], value=0.0)
    m5: unreal.Map[str, float] = unreal.Map.fromkeys(("A", "B", "C"), 0.0)
    m6: unreal.Map[str, float] = unreal.Map.fromkeys({"A": 0, "B": 1, "C": 2}, 0.0)  # From the keys of a dict.
    local_print(m4)
    local_print(m5)
    local_print(m6)

    # get(self, key: KT, default: VT = ...) -> VT:
    value: str = m.get(0)
    local_print(value)
    value: str = m.get(key=0, default="bar")
    local_print(value)

    # setdefault(self, key: KT, default: VT = ...) -> VT:
    value: str = m.setdefault(99)
    local_print(value)
    value: str = m.setdefault(key=44, default="foo")
    local_print(value)

    # pop(self, key: KT, default: VT = ...) -> VT:
    value: str = m.pop(99)
    local_print(value)
    value: str = m.pop(99, "joe")
    local_print(value)
    value: str = m.pop(key=99, default="joe")
    local_print(value)

    # popitem(self) -> tuple[KT, KV]:
    item: tuple[int, str] = m.popitem()
    local_print(item)

    # update(self, pairs: Union[Iterable[Any], Mapping[KT, VT]]) -> None:
    m.update([(10, "A"), (20, "B")])  # Iterable of tuples
    m.update([[30, "C"], [40, "D"]])  # Iterable of 2-element list 
    m.update({50: "E", 60: "F"})      # Mapping of int, str.
    m.update(m2)                      # Map[int, str]
    
    # items(self) -> ItemsView[KT, VT]:
    items: unreal.ItemsView[int, str] = m.items()
    for i in items:
        local_print(i)

    # keys(self) -> Iterable[KT]:
    keys: typing.Iterable[int] = m.keys()
    for k in keys:
        local_print(k)

    # values(self) -> Iterable[VT]:
    values: typing.Iterable[str] = m.values()
    for v in values:
        local_print(v)
    
    # Check polymorphism.
    poly: unreal.Map[typing.Union[unreal.Name, str], unreal.Object] = unreal.Map(unreal.Name, unreal.Object)
    poly["joe"] = unreal.Actor()  # Accept "joe" because a 'str' is convertible to a 'unreal.Name'

    # Check using Optional and None.
    int_obj_map: unreal.Map[int, typing.Optional[unreal.Object]] = unreal.Map(int, unreal.Object)
    int_obj_map.__setitem__(0, None)
    int_obj_map[10] = None
    
    # Type coercion
    name_int_map: unreal.Map[typing.Union[str, unreal.Name], int] = unreal.Map(unreal.Name, int)
    name_int_map.__setitem__("hello", 1)
    name_int_map["hello"] = 10


def test_reflected_types() -> None:
    """ Ensures that UE reflected types are correctly hinted (from reflection). """

    # Check for init method (Using a PyTestStruct because PyTestTypeHint uses the init from base UObject)
    s: unreal.PyTestStruct = unreal.PyTestStruct(False, 0, 0.0, unreal.PyTestEnum.ONE, 
        "Str", "Name", "Text", unreal.FieldPath(), unreal.FieldPath(), ["StrArray"], {"StrSet"}, {"StrIntMap": 1})
    local_print(s)

    o = unreal.PyTestTypeHint()
    str_const: str = unreal.PyTestTypeHint.STR_CONST
    int_const: int = unreal.PyTestTypeHint.INT_CONST
    local_print(str_const)
    local_print(int_const)

    bool_prop: bool = o.bool_prop
    bool_retv: bool = o.check_bool_type_hints(bool_prop)
    local_print(bool_retv)

    int_prop: int = o.int_prop
    int_retv: int = o.check_integer_type_hints(int_prop)
    local_print(int_retv)

    float_prop: float = o.float_prop
    float_retv: float = o.check_float_type_hints(float_prop, 0.0)
    local_print(float_retv)

    enum_prop: unreal.PyTestEnum = o.enum_prop
    local_print(enum_prop)
    enum_prop = unreal.PyTestEnum.ONE
    enum_retv: unreal.PyTestEnum = o.check_enum_type_hints(enum_prop)
    local_print(enum_retv)

    str_prop: str = o.string_prop
    str_retv: str = o.check_string_type_hints(str_prop)
    local_print(str_retv)

    name_prop: unreal.Name = o.name_prop
    name_retv: unreal.Name = o.check_name_type_hints(name_prop)
    local_print(name_retv)
    o.name_prop = "some str"  # Type coercion from str to unreal.Name()
    o.check_name_type_hints("Hi")  # Type coercion from str to unreal.Name()

    text_prop: unreal.Text = o.text_prop
    text_retv: unreal.Text = o.check_text_type_hints(text_prop)
    local_print(text_retv)
    o.text_prop = "some str"  # Type coercion from str to unreal.Text()
    o.check_text_type_hints("Hi")  # Type coercion from str to unreal.Text()

    field_path_prop: unreal.FieldPath = o.field_path_prop
    field_path_retv: unreal.FieldPath = o.check_field_path_type_hints(field_path_prop)
    local_print(field_path_retv)

    struct_prop: unreal.PyTestStruct = o.struct_prop
    struct_retv: unreal.PyTestStruct = o.check_struct_type_hints(struct_prop)
    unreal.PyTestObject().func_taking_py_test_struct([True])  # List can be coerced into struct
    unreal.PyTestObject().func_taking_py_test_struct({"bool": True})  # Dict can be coerced into struct
    unreal.PyTestObject().func_taking_py_test_struct({"bool": True, "int": 44, "float": 44.5}.values())  # Iterable can be coerced into struct
    local_print(struct_retv)

    object_prop: typing.Optional[unreal.PyTestObject] = o.object_prop
    object_retv: typing.Optional[unreal.PyTestObject] = o.check_object_type_hints(object_prop)
    o.object_prop = None  # Property can be None.
    o.object_prop = unreal.PyTestChildObject()  # Can be a derived type.
    local_print(object_retv)

    # Native Python data structure to test unreal.Array(), unreal.Set() and unreal.Map() function parameters.
    py_str_list: typing.List[str] = ["a", "b", "c"]
    py_str_set: typing.Set[str] = {"a", "b", "c"}
    py_str_tuple: typing.Tuple[str, str, str] = ("a", "b", "c")
    py_str_int_dict: typing.Mapping[str, int] = {"a": 0, "b": 1, "c": 2}
    py_int_str_dict: typing.Mapping[int, str] = {1: "a", 2: "b", 3: "c"}
    py_obj_list: typing.List[unreal.PyTestObject] = [unreal.PyTestObject()]
    py_obj_set: typing.Set[unreal.PyTestObject] = {unreal.PyTestObject()}
    py_int_obj_dict: typing.Mapping[int, unreal.PyTestObject] = {0: unreal.PyTestObject(), 1: unreal.PyTestObject()}

    string_array: unreal.Array[str] = o.str_array_prop
    name_array: unreal.Array[unreal.Name] = unreal.Array(unreal.Name)
    text_array: unreal.Array[unreal.Text] = unreal.Array(unreal.Text)
    object_array: unreal.Array[unreal.Object] = unreal.Array(unreal.Object)
    array_retv: unreal.Array[unreal.Text] = o.check_array_type_hints(string_array, name_array, text_array, object_array)
    local_print(array_retv)
    # Array[Name] requires Name obj. Ideally Array[T] API would coerce string into Name, but that doesn't look feasible.
    o.name_array_prop.append(unreal.Name("foo"))
    o.name_array_prop = name_array
    # Array[Text] requires Text obj. Ideally Array[T] API would coerce string into Text, but that doesn't look feasible.
    o.text_array_prop.append(unreal.Text("foo"))
    o.text_array_prop = text_array
    # Array[Object] property accepts None as element.
    o.object_array_prop.append(None)
    # Accepts a Python Tuple[] in place of unreal.Array[] + type coercion of str in place of Name/Text + polymorphism
    o.check_array_type_hints(py_str_tuple, py_str_tuple, py_str_tuple, py_obj_list)
    # Accepts a Python List[] in place of unreal.Array[] + type coercion of str in place of Name/Text + polymorphism
    o.check_array_type_hints(py_str_list, py_str_list, py_str_list, py_obj_list)
    # Accepts a Python Iterable[] in place of unreal.Array[] + type coercion of str in place of Name/Text + polymorphism
    o.check_array_type_hints(py_str_int_dict.keys(), py_str_int_dict.keys(), py_str_int_dict.keys(), py_int_obj_dict.values())
    # Accepts a Python Set[] in place of an unreal.Array[] + type coercion of str in place of Name/Text + polymorphism
    o.check_array_type_hints(py_str_set, py_str_set, py_str_set, py_obj_set)
    # Accepts empty Python data structures
    o.check_array_type_hints([], set(), [], [])

    string_set: unreal.Set[str] = o.set_prop
    name_set: unreal.Set[unreal.Name] = unreal.Set(unreal.Name)
    object_set: unreal.Set[unreal.Object] = unreal.Set(unreal.Object)
    set_retv: unreal.Set[unreal.Name] = o.check_set_type_hints(string_set, name_set, object_set)
    local_print(set_retv)
    # Accepts a Python Set[] in place of unreal.Set[] + type coercion of str in place of Name + polymorphism
    o.check_set_type_hints(py_str_set, py_str_set, py_obj_set)
    # Accepts a Python List[] in place of unreal.Set[] + type coercion of str in place of Name + polymorphism
    o.check_set_type_hints(py_str_list, py_str_list, py_obj_list)
    # Accepts a Python Iterable[] in place of unreal.Set[] + type coercion of str in place of Name + polymorphism
    o.check_set_type_hints(py_str_int_dict.keys(), py_str_int_dict.keys(), py_int_obj_dict.values())
    # Accepts empty Python data structures
    o.check_set_type_hints([], set(), [])

    int_str_map: unreal.Map[int, str] = o.map_prop
    int_name_map: unreal.Map[int, unreal.Name] = unreal.Map(int, unreal.Name)
    int_text_map: unreal.Map[int, unreal.Text] = unreal.Map(int, unreal.Text)
    int_obj_map: unreal.Map[int, unreal.Object] = unreal.Map(int, unreal.Object)
    map_retv: unreal.Map[str, typing.Optional[unreal.Object]] = o.check_map_type_hints(int_str_map, int_name_map, int_text_map, int_obj_map)
    local_print(map_retv)
    # Accepts a Python Dict[] in place of unreal.Map[] + type coercion of str in place of Name/Text + polymorphism
    o.check_map_type_hints(py_int_str_dict, py_int_str_dict, py_int_str_dict, py_int_obj_dict)
    # Accepts a list of tuple + type coercion of str in place of Name/Text + polymorphism
    o.check_map_type_hints([(1, "A"), (2, "B")], [(1, "A"), (2, "B")], [(1, "A"), (2, "B")], py_int_obj_dict)
    # Accepts a list of 2-element list + type coercion of str in place of Name/Text + polymorphism
    o.check_map_type_hints([[1, "A"], [2, "B"]], [[1, "A"], [2, "B"]], [[1, "A"], [2, "B"]], py_int_obj_dict)
    # Accepts empty Python data structures
    o.check_map_type_hints({}, {}, {}, {})
    o.check_map_type_hints([[1, "A"], [2, "B"]], [[1, "A"], [2, "B"]], [[1, "A"], [2, "B"]], {2: None})

    delegate_callable = DelegateCallable()
    delegate_prop: unreal.PyTestDelegate = o.delegate_prop
    delegate_retv: unreal.PyTestDelegate = o.check_delegate_type_hints(delegate_prop)
    if not delegate_retv.is_bound():
        delegate_prop.bind_callable(delegate_callable.delegate_callback)
    delegate_prop.unbind()

    multicast_delegate_prop: unreal.PyTestMulticastDelegate = o.multicast_delegate_prop
    o.multicast_delegate_prop = unreal.PyTestMulticastDelegate()
    o.multicast_delegate_prop = multicast_delegate_prop

    bool_retv: bool = unreal.PyTestTypeHint.check_static_function(True, 0, 0.1, "")
    tuple_retv: typing.Tuple[int, str] = unreal.PyTestTypeHint.check_tuple_return_type("foo")
    local_print(bool_retv)
    local_print(tuple_retv)
    

def test_core_module() -> None:
    """ 
    This function is here to remind that some types are defines in unreal_core.py and are
    pulled in the stub file, for example uclass(), ustruct(), uenum(), uvalue(), uproperty(),
    ufunction() are methods defined in that file. They are not hinted yet because hint
    could not be turned off. Ideally, that file will be manually hinted once we set a minimum
    Python version of 3.9. Currently, user are free to recompile the engine against a 3.x
    version.
    """
    pass


def test_slow_task() -> None:
    """ Ensure the manually hinted SlowTask API is correctly hinted (In PyCore.cpp). """
    total_work: int = 5
    work_per_frame: float = 1.0
    task: unreal.ScopedSlowTask
    with unreal.ScopedSlowTask(work=total_work, desc="Testing Hinting") as task:
        task.make_dialog(can_cancel=True, allow_in_pie=False)
        for _ in range(total_work):
            if task.should_cancel():
                break
            task.enter_progress_frame(work_per_frame, "Doing some work...")

    with unreal.ScopedSlowTask(total_work, "Testing Hinting") as task:
        task.make_dialog_delayed(1.0, can_cancel=True, allow_in_pie=False)
        for _ in range(total_work):
            if task.should_cancel():
                break
            task.enter_progress_frame(work=work_per_frame, desc="Doing some work...")


def test_py_core_methods() -> None:
    """ Ensure the methods manually hinted in PyCore.cpp are correctly hinted. """

    # log(arg: Any) -> None
    unreal.log(0)
    unreal.log("msg")
    unreal.log((0, 1, 2))

    # log_warning(arg: Any) -> None
    unreal.log_warning(1)
    unreal.log_warning("msg")
    unreal.log_warning([0, 1, 2])

    # log_error(arg: Any) -> None
    unreal.AutomationLibrary.add_expected_log_error(str(42), 1, True)
    unreal.log_error(42)
    unreal.AutomationLibrary.add_expected_log_error("msg", 1, True)
    unreal.log_error("msg")
    unreal.AutomationLibrary.add_expected_log_error(".a.: 10, .b.: 20", 1, False)
    unreal.log_error({"a": 10, "b": 20})

    # log_flush() -> None
    unreal.log_flush()

    # reload()/load_module()
    module_name = "core"  # This will load/reload "unreal_core.py"
    unreal.reload(module_name)
    unreal.load_module(module_name)
    
    # new_object
    valid_obj: unreal.PyTestObject = unreal.new_object(unreal.PyTestObject.static_class(), None, "MyObject", None)
    local_print(valid_obj)

    # find_object()/load_object() - The specified asset may not exist, but type match and would work if the Blueprint asset existed.
    loaded_object: typing.Optional[unreal.Object] = unreal.load_object(None, "/Game/AAAA/ActorBP.ActorBP", unreal.Object.static_class())  # outer = None -> transient package.
    local_print(loaded_object)
    found_object: typing.Optional[unreal.Object] = unreal.find_object(None, "/Game/AAAA/ActorBP.ActorBP", unreal.Object.static_class())  # outer = None -> transient package.
    local_print(found_object)

    # load_class() - The specified class may not exist, but types match and this would work if Blueprint asset existed.
    loaded_class: typing.Optional[unreal.Class] = unreal.load_class(None, "Blueprint'/Game/AAAA/ActorBP.ActorBP_C'")  # outer = None -> transient package.
    local_print(loaded_class)

    # find_asset()/load_asset() - The specified asset doesn't exist... it just to test type hinting.
    loaded_asset: typing.Optional[unreal.Blueprint] = unreal.load_asset("/Game/AAAA/ActorBP.ActorBP", unreal.Blueprint.static_class(), follow_redirectors=True)
    local_print(loaded_asset)
    found_asset: typing.Optional[unreal.Blueprint] = unreal.find_asset("/Game/AAAA/ActorBP.ActorBP", unreal.Blueprint.static_class(), follow_redirectors=True)
    local_print(found_asset)

    # find_package()/load_package()
    loaded_pkg: typing.Optional[unreal.Package] = unreal.load_package("/Game/AAAA/ActorBP")
    local_print(loaded_pkg)
    found_pkg: typing.Optional[unreal.Package] = unreal.find_package("/Game/AAAA/ActorBP")
    local_print(found_pkg)

    # get_default_object()
    cdo: typing.Optional[unreal.PyTestObject] = unreal.get_default_object(unreal.PyTestObject.static_class())
    local_print(cdo)

    # purge_object_references()
    unreal.purge_object_references(unreal.PyTestObject(), include_inners=True)

    # generate_class()/generate_struct()/generate_enum() -> Those are used by @unreal.uenum, @unreal.ustruct and @unreal.uclass decorator
    # that are defined in unreal_core.py module. Normally, user shouldn't need to call them, but type checked.
    unreal.generate_enum(unreal.PyTestEnum)
    unreal.generate_struct(unreal.PyTestStruct)
    unreal.generate_class(unreal.PyTestObject)

    # get_type_from_class()/get_type_from_struct()/get_type_from_enum()
    cls_type: type = unreal.get_type_from_class(unreal.PyTestObject.static_class())
    local_print(cls_type)
    struct_type: type = unreal.get_type_from_struct(unreal.PyTestStruct.static_struct())
    local_print(struct_type)
    enum_type: type = unreal.get_type_from_enum(unreal.PyTestEnum.static_enum())
    local_print(enum_type)

    # register_python_shutdown_callback()/unregister_python_shutdown_callback()
    def shutdown_callback() -> None:
        local_print("goodbye!")
    opaque_handle = unreal.register_python_shutdown_callback(shutdown_callback)
    unreal.unregister_python_shutdown_callback(opaque_handle)

    # NSLOCTEXT/LOCTABLE -> 'StrTable' is a 'String Table' asset that can be created in the Content Browser.
    found_text: unreal.Text = unreal.LOCTABLE("/Game/AAAA/StrTable", "Foo")
    local_print(found_text)
    loc_text: unreal.Text = unreal.NSLOCTEXT("MyNamespace", "NewKey2", "NewKeyValue")
    local_print(loc_text)

    # is_editor()
    editor_runtime: bool = unreal.is_editor()
    local_print(editor_runtime)

    # get_interpreter_executable_path()
    path: str = unreal.get_interpreter_executable_path()
    local_print(path)

    # Object iterator (passing a type)
    object_it = unreal.ObjectIterator(unreal.PyTestObject)
    visited_obj: unreal.PyTestObject
    for visited_obj in object_it:
        local_print(visited_obj)
    
    # Object iterator (passing a unreal.Class)
    static_mesh_it = unreal.ObjectIterator(unreal.StaticMesh.static_class())
    visited_mesh_object: unreal.StaticMesh
    for visited_mesh_object in static_mesh_it:
        local_print(visited_mesh_object)

    # Class iterator
    class_it = unreal.ClassIterator(unreal.StaticMeshActor)
    visited_class: unreal.Class
    for visited_class in class_it:
        local_print(visited_class)

    # Struct iterator
    struct_it = unreal.StructIterator(unreal.PyTestStruct)
    visited_script_struct: unreal.ScriptStruct
    for visited_script_struct in struct_it:
        local_print(visited_script_struct)
    
    # Type iterator
    type_it = unreal.TypeIterator(unreal.PyTestObject)
    visited_type: type
    for visited_type in type_it:
        local_print(visited_type)


def test_py_editor_methods() -> None:
    """ Ensure the methods manually hinted in PyEditor.cpp are correctly hinted. """
    engine_ss1: typing.Optional[unreal.EngineSubsystem] = unreal.get_engine_subsystem(unreal.SubobjectDataSubsystem)  # Using type
    local_print(engine_ss1)
    engine_ss2: typing.Optional[unreal.EngineSubsystem] = unreal.get_engine_subsystem(unreal.SubobjectDataSubsystem.static_class())  # Using Class
    local_print(engine_ss2)
    editor_ss1: typing.Optional[unreal.EditorSubsystem] = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)  # Using type.
    local_print(editor_ss1)
    editor_ss2: typing.Optional[unreal.EditorSubsystem] = unreal.get_editor_subsystem(unreal.EditorActorSubsystem.static_class())  # Using Class
    local_print(editor_ss2)

    with unreal.ScopedEditorTransaction(desc="My Transaction"):
        local_print("do something")


def test_py_slate_methods() -> None:
    def tick_callable(dt: float) -> None:
        local_print(dt)

    # Using a Python callable
    pre_tick_handle: object = unreal.register_slate_pre_tick_callback(tick_callable)
    post_tick_handle: object = unreal.register_slate_post_tick_callback(tick_callable)
    unreal.unregister_slate_pre_tick_callback(pre_tick_handle)
    unreal.unregister_slate_post_tick_callback(post_tick_handle)

    # Using an unreal delegate.
    o = unreal.PyTestTypeHint()
    o.slate_tick_delegate.bind_callable(tick_callable)
    pre_tick_handle = unreal.register_slate_pre_tick_callback(o.slate_tick_delegate)
    post_tick_handle = unreal.register_slate_post_tick_callback(o.slate_tick_delegate)
    unreal.unregister_slate_pre_tick_callback(pre_tick_handle)
    unreal.unregister_slate_post_tick_callback(post_tick_handle)
    o.slate_tick_delegate.unbind()

    # Always false. The enclosed instructions are type-verified, but never executed.
    if o.slate_tick_delegate.is_bound():
        opaque_window_handle: object = 0  # Using an int, but this could be any other object type.
        unreal.parent_external_window_to_slate(opaque_window_handle, unreal.SlateParentWindowSearchMethod.ACTIVE_WINDOW)


def test_py_engine_methods() -> None:
    # The function 'get_blueprint_generated_types' accepts user-defined 'Enumeration' asset which can be created from the content browser.
    asset_subsystem : typing.Optional[unreal.EditorAssetSubsystem] = unreal.get_editor_subsystem(unreal.EditorAssetSubsystem)
    if asset_subsystem is not None and asset_subsystem.does_asset_exist("/Game/AAAA/MyEnum"):
        from_str = unreal.get_blueprint_generated_types("/Game/AAAA/MyEnum")
        local_print(from_str)

        from_list = unreal.get_blueprint_generated_types(["/Game/AAAA/MyEnum"])
        local_print(from_list)

        from_set = unreal.get_blueprint_generated_types({"/Game/AAAA/MyEnum"})
        local_print(from_set)

        from_tuple = unreal.get_blueprint_generated_types(("/Game/AAAA/MyEnum",))
        local_print(from_tuple)

        from_iterable = unreal.get_blueprint_generated_types({"/Game/AAAA/MyEnum": 0}.keys())
        local_print(from_iterable)

    # Iterate all actor (testing default param)
    editor_subsystem: typing.Optional[unreal.UnrealEditorSubsystem] = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem)
    if editor_subsystem is not None:
        world: typing.Optional[unreal.World] = editor_subsystem.get_editor_world()
        if world:
            actor_it: unreal.ActorIterator = unreal.ActorIterator(world)
            for actor in actor_it:
                local_print(actor)

            # Iterate a specific type of actor.
            selected_actor_it: unreal.SelectedActorIterator = unreal.SelectedActorIterator(world, unreal.StaticMeshActor)
            static_mesh_actor: unreal.StaticMeshActor
            for static_mesh_actor in selected_actor_it:
                local_print(static_mesh_actor)

    unreal.ValueDef(3, {"DisplayName": "Simple Value"})
    unreal.PropertyDef(int, {"DisplayName": "Simple Prop"}, None, None)
    unreal.FunctionDef(unreal.log, {"DisplayName": "Simple Func"}, None, None, None, None, None, None, None)


if __name__ == "__main__":
    test_name_wrapper()
    test_text_wrapper()
    test_delegate_wrapper()
    test_multicast_delegate_wrapper()
    test_field_type_wrapper()
    test_enum_wrapper()
    test_struct_wrapper()
    test_object_wrapper()
    test_fixed_array_wrapper()
    test_array_wrapper()
    test_set_wrapper()
    test_map_wrapper()
    test_reflected_types()
    test_core_module()
    test_slow_task()
    test_py_core_methods()
    test_py_editor_methods()
    test_py_slate_methods()
    test_py_engine_methods()
    
