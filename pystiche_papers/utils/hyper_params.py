import copy
import itertools
from collections import OrderedDict
from collections.abc import Mapping
from typing import Any, Dict, Iterator, Optional, Tuple

import pystiche

__all__ = ["HyperParameters"]


class HyperParameters(Mapping, pystiche.ComplexObject):
    def __init__(self, **kwargs: Any) -> None:
        self.__params__: Dict[str, Any] = OrderedDict()
        self.__sub_params__: Dict[str, "HyperParameters"] = OrderedDict()

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

    def __getitem__(self, name: str) -> Any:
        return getattr(self, name)

    def __iter__(self) -> Iterator:
        return itertools.chain(self.__params__.keys(), self.__sub_params__.keys())

    def __len__(self) -> int:
        return len(self.__params__) + len(self.__sub_params__)

    def __copy__(self) -> "HyperParameters":
        params = copy.copy(self.__params__)
        params.update(
            {
                name: copy.copy(sub_param)
                for name, sub_param in self.__sub_params__.items()
            }
        )
        return type(self)(**params)

    def __deepcopy__(
        self,
        memo: Optional[Dict[int, Any]] = None,
    ) -> "HyperParameters":
        params = copy.deepcopy(self.__params__, memo=memo)
        params.update(
            {
                name: copy.deepcopy(sub_param, memo=memo)
                for name, sub_param in self.__sub_params__.items()
            }
        )
        return type(self)(**params)

    def new_similar(self, deepcopy: bool = False, **kwargs: Any) -> "HyperParameters":
        hyper_parameters = copy.deepcopy(self) if deepcopy else copy.copy(self)
        for name, value in kwargs.items():
            setattr(hyper_parameters, name, value)
        return hyper_parameters

    def _properties(self) -> Dict[str, Any]:
        dct = super()._properties()
        dct.update(self.__params__)
        return dct

    def _named_children(self) -> Iterator[Tuple[str, Any]]:
        yield from self.__sub_params__.items()
