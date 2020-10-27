import pytest

__all__ = ["data", "impl_params"]


def data(argnames, argvalues, ids=None, **kwargs):
    if isinstance(argnames, str):
        argnames = [name.strip() for name in argnames.split(",")]

    single_arg = len(argnames) == 1

    if ids is None:
        if single_arg:
            ids = [f"{argnames[0]}={value}" for value in argvalues]
        else:
            ids = [
                ", ".join(f"{name}={value}" for name, value in zip(argnames, values))
                for values in argvalues
            ]

    return pytest.mark.parametrize(
        argnames[0] if single_arg else argnames, argvalues, ids=ids, **kwargs
    )


impl_params = data("impl_params", (True, False))
