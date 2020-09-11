__all__ = ["assert_property_in_repr"]


def assert_property_in_repr(repr, name, value):
    assert f"{name}={value}" in repr
