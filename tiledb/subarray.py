from numbers import Real
from typing import Sequence, Tuple, Union

import numpy as np

import tiledb.cc as lt

from .ctx import Ctx, CtxMixin, default_ctx
from .libtiledb import Array

Scalar = Real
Range = Tuple[Scalar, Scalar]


class Subarray(CtxMixin, lt.Subarray):
    """
    Represents a TileDB subarray.
    """

    def __init__(
        self,
        array: Array,
        ctx: Ctx = None,
    ):
        """Class representing a subarray of a TileDB Array.

        :param array: tiledb.Array the subarray is defined on
        :param ctx: A TileDB Context
        """
        self._array = array
        super().__init__(
            ctx, lt.Array(ctx if ctx is not None else default_ctx(), array)
        )

    def add_dim_range(self, dim_idx: int, dim_range: Range):
        """Add a range to a dimension of the subarray.

        :param dim_idx: dimension index (int) of the dimension to add the range to
        :param range: tuple containing the inclusive range to query on
        :raises: :py:exc:`tiledb.TileDBError`
        """
        self._add_dim_range(dim_idx, dim_range)

    def add_label_range(self, label: str, label_range: Range):
        """Add a dimension label to a dimension of the subarray.

        :param label: name (str) of the label to add a range to
        :param range: tuple containing the inclusive range to query on
        :raises: :py:exc:`tiledb.TileDBError`
        """
        self._add_label_range(self._ctx, label, label_range)

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

    def num_dim_ranges(self, key: Union[int, str]) -> np.uint64:
        """Returns the number of ranges on a dimension.

        :param key: dimension index (int) or name (str)
        :rtype: np.uint64
        """
        return self._range_num(key)

    def num_label_ranges(self, label: str) -> np.uint64:
        """Returns the number of ranges on a dimension label.

        :param key: dimensio label name
        :rtype: np.uint64
        """
        return self._range_label_num(label)
