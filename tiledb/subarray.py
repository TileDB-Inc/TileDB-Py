from numbers import Real
from typing import Dict, Optional, Sequence, Tuple, Union

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

    def add_ranges(
        self,
        dim_ranges: Optional[Sequence[Sequence[Range]]] = None,
        label_ranges: Optional[Dict[str, Sequence[Range]]] = None,
    ):
        """Add ranges to the subarray.

        :param dim_ranges: A sequence of a sequence of ranges to add for each dimension
            on the subarray. Each range may either be a tuple (inclusive range to query
            on) or  numpy.ndarray (series of point ranges).
        :param label_ranges: A dictionary of label name to a sequence of ranges for the
            dimension label on the subarray. Each range must be a tuple (inclusive range
            to query on).
        :raises: :py:exc:`tiledb.TileDBError`
        """
        if dim_ranges:
            if any(isinstance(r, np.ndarray) for r in dim_ranges):
                self._add_ranges_bulk(self._ctx, dim_ranges)
            else:
                self._add_ranges(self._ctx, dim_ranges)
        if label_ranges:
            self._add_label_ranges(self._ctx, label_ranges)

    def has_label_range(self, dim_idx):
        """Returns if dimension label ranges are set on the requested dimension.

        :param dim_idx: Index (int) of the dimension to check for labels.
        :rtype: int
        """
        return self._has_label_range(self._ctx, dim_idx)

    def num_dim_ranges(self, key: Union[int, str]) -> np.uint64:
        """Returns the number of ranges on a dimension.

        :param key: dimension index (int) or name (str)
        :rtype: np.uint64
        """
        return self._range_num(key)

    def num_label_ranges(self, label: str) -> np.uint64:
        """Returns the number of ranges on a dimension label.

        :param key: dimension label name
        :rtype: np.uint64
        """
        if not isinstance(label, str):
            raise TypeError(f"invalid type {type(label)} for label")
        return self._label_range_num(self._ctx, label)

    def shape(self) -> Optional[Sequence[tuple]]:
        """Returns the shape of dense data using this subarray and ``None`` for sparse
        arrays.

        Warning: This is the shape of time dimension ranges. If a label range is set
        and the dimension range has not been computed, it will return `0`.

        :rtype: tuple(int, ...)
        :raises: :py:exc:`tiledb.TileDBError`
        """
        if self._array.schema.sparse:
            return None
        shape = self._shape(self._ctx)
        return tuple(length for length in shape)
