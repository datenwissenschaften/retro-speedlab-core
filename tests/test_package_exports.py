import pytest

import datenwissenschaften


def test_every_declared_export_resolves():
    for name in datenwissenschaften.__all__:
        assert getattr(datenwissenschaften, name) is not None


def test_unknown_attribute_raises_attribute_error():
    with pytest.raises(AttributeError):
        datenwissenschaften.does_not_exist
