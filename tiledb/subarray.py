from typing import (
    TYPE_CHECKING,
    Iterator,
    Optional,
    Sequence,
    Tuple,
    List,
    Any,
    Union,
    cast,
)
import numpy as np
from numbers import Real

import tiledb.cc as lt
from .libtiledb import Array
from .ctx import CtxMixin
from ._array import ArrayImpl

if TYPE_CHECKING:
    from libtiledb import Ctx


Scalar = Real
Range = Tuple[Scalar, Scalar]


class Subarray(CtxMixin, lt.Subarray):
    """
    TODO: Documentation for subarray
    """

    def __init__(
        self,
        array: Array,
        ctx: "Ctx" = None,
    ):
        """TODO: Documentation"""
        # Initialize pybind Subarray
        self._array = array
        self._cpp_array = ArrayImpl(array)
        super().__init__(ctx, self._cpp_array)

    def nrange(self, key) -> np.uint64:
        """TODO: Documentation"""
        return self._range_num(key)

    def add_ranges(self, ranges: Sequence[Sequence[Range]]):
        """TODO: Documentation"""
        if any(isinstance(r, np.ndarray) for r in ranges):
            self._add_ranges_bulk(self._ctx, ranges)
        else:
            self._add_ranges(self._ctx, ranges)

    def add_dim_range(self, dim_idx: int, range: Range) -> None:
        """TODO: Documentation"""
        self._add_dim_range(dim_idx, range)

    def add_label_range(self, label: str, range: Range):
        """TODO: Documentation"""
        self._add_label_range(self, self._ctx, label, range)
