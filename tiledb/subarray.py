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
    Represents a TileDB subarray.
    """

    def __init__(
        self,
        array: Array,
        ctx: "Ctx" = None,
    ):
        """Class representing a subarray of a TileDB Array.

        :param array: tiledb.Array the subarray is defined on
        :param ctx: A TileDB Context
        """
        self._array = array
        self._cpp_array = ArrayImpl(array)
        super().__init__(ctx, self._cpp_array)

    def nrange(self, key) -> np.uint64:
        """Returns the number of ranges on a dimension.

        :param key: dimension index (int) or name (str)
        :rtype: np.uint64
        """
        return self._range_num(key)

    def add_ranges(self, ranges: Sequence[Sequence[Range]]):
        """Add ranges to the subarray.

        :param ranges: A sequence of a sequence of ranges to add for each dimension on
            the subarray. Each range may either be a tuple (inclusive range to query
            on) or  numpy.ndarray (series of point ranges).
        :raises: :py:exc:`tiledb.TileDBError`
        """
        if any(isinstance(r, np.ndarray) for r in ranges):
            self._add_ranges_bulk(self._ctx, ranges)
        else:
            self._add_ranges(self._ctx, ranges)

    def add_dim_range(self, dim_idx: int, range: Range):
        """Add a range to a dimension of the subarray.

        :param dim_idx: dimension index (int) of the dimension to add the range to
        :param range: tuple containing the inclusive range to query on
        :raises: :py:exc:`tiledb.TileDBError`
        """
        self._add_dim_range(dim_idx, range)

    def add_label_range(self, label: str, range: Range):
        """Add a dimension label to a dimension of the subarray.

        :param label: name (str) of the label to add a range to
        :param range: tuple containing the inclusive range to query on
        :raises: :py:exc:`tiledb.TileDBError`
        """
        self._add_label_range(self, self._ctx, label, range)
