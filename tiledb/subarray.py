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
            # TODO: Before merging move this to C++
            for label_name, selection in label_ranges.items():
                for rng in selection:
                    self._add_label_range(self._ctx, label_name, rng)

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
