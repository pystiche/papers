from collections import OrderedDict
from typing import Any, Dict, Iterator, Tuple

import pystiche

__all__ = ["HyperParameters"]


class HyperParameters(pystiche.ComplexObject):
    def __init__(self, **kwargs: Any) -> None:
        self.__params__: Dict[str, Any] = OrderedDict()
        self.__sub_params__: Dict[str, Any] = OrderedDict()

        for name, value in kwargs.items():
            setattr(self, name, value)

    def __getattr__(self, name: str) -> Any:
        for dct in (self.__params__, self.__sub_params__):
            if name in dct:
                return dct[name]
        else:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

    def __setattr__(self, name: str, value: Any) -> None:
        if name in ("__params__", "__sub_params__"):
            super().__setattr__(name, value)
            return

        new_dct, old_dct = (
            (self.__sub_params__, self.__params__)
            if isinstance(value, HyperParameters)
            else (self.__params__, self.__sub_params__)
        )

        if name in old_dct:
            del old_dct[name]
        new_dct[name] = value

    def __delattr__(self, name: str) -> None:
        for dct in (self.__params__, self.__sub_params__):
            if name in dct:
                del dct[name]
                break
        else:
            super().__delattr__(name)

    def __contains__(self, name: str) -> bool:
        return name in self.__params__ or name in self.__sub_params__

    def _properties(self) -> Dict[str, Any]:
        dct = super()._properties()
        dct.update(self.__params__)
        return dct

    def _named_children(self) -> Iterator[Tuple[str, Any]]:
        yield from self.__sub_params__.items()
