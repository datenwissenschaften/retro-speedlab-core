from dataclasses import dataclass
from pathlib import Path

from datenwissenschaften.serialization import to_json_value


@dataclass
class _Point:
    x: int
    y: int


def test_primitives_pass_through():
    assert to_json_value(None) is None
    assert to_json_value("text") == "text"
    assert to_json_value(True) is True
    assert to_json_value(5) == 5


def test_finite_float_passes_through():
    assert to_json_value(1.5) == 1.5


def test_non_finite_float_is_stringified():
    assert to_json_value(float("inf")) == "inf"
    assert to_json_value(float("nan")) == "nan"


def test_path_is_stringified():
    assert to_json_value(Path("a/b")) == str(Path("a/b"))


def test_dataclass_instance_is_converted_recursively():
    assert to_json_value(_Point(1, 2)) == {"x": 1, "y": 2}


def test_dataclass_type_itself_is_treated_as_a_plain_callable():
    # `_Point` (the class) is itself a dataclass, but `is_dataclass` is True
    # for both instances and types; the `not isinstance(value, type)` guard
    # sends the type itself down the callable path instead.
    assert to_json_value(_Point) == "_Point"


def test_dict_keys_are_stringified_and_values_converted():
    assert to_json_value({1: Path("a")}) == {"1": str(Path("a"))}


def test_lists_and_tuples_are_converted_to_lists():
    assert to_json_value([1, Path("a")]) == [1, str(Path("a"))]
    assert to_json_value((1, 2)) == [1, 2]


def test_sets_are_converted_to_lists():
    assert sorted(to_json_value({1, 2})) == [1, 2]


def test_named_callables_report_their_name():
    def sample():
        pass

    assert to_json_value(sample) == "sample"


def test_callables_without_a_name_fall_back_to_str():
    class _Callable:
        def __call__(self):
            return None

        def __repr__(self):
            return "callable-repr"

    assert to_json_value(_Callable()) == "callable-repr"


def test_unknown_objects_fall_back_to_str():
    class _Custom:
        def __repr__(self):
            return "custom-repr"

    assert to_json_value(_Custom()) == "custom-repr"
