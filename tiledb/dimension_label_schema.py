from enum import Enum
from typing import Any, Sequence, Union

import numpy as np

import tiledb.cc as lt

from .ctx import Ctx, default_ctx
from .filter import Filter, FilterList
from .util import dtype_to_tiledb, numpy_dtype, tiledb_cast_tile_extent


class DataOrder(Enum):
    increasing = lt.DataOrder.INCREASING_DATA
    decreasing = lt.DataOrder.DECREASING_DATA
    unordered = lt.DataOrder.UNORDERED_DATA


class DimLabelSchema:
    def __init__(
        self,
        dim_index: np.uint32,
        order: str = "increasing",
        label_dtype: np.dtype = np.uint64,
        dim_dtype: np.dtype = np.uint64,
        dim_tile: Any = None,
        label_filters: Union[FilterList, Sequence[Filter]] = None,
        ctx: Ctx = None,
    ):
        # Set context
        self._ctx = ctx or default_ctx()

        # Set simple properties
        self._dim_index = dim_index
        self._label_order = DataOrder[order]

        # Compute and set the label datatype
        if isinstance(label_dtype, str) and label_dtype == "ascii":
            self._label_dtype = lt.DataType.STRING_ASCII
        else:
            label_dtype = np.dtype(label_dtype)
            if label_dtype.kind == "S":
                self._label_dtype = lt.DataType.STRING_ASCII
            self._label_dtype = dtype_to_tiledb(label_dtype)

        # Compute and set the dimension datatype and filter
        if isinstance(dim_dtype, str) and dim_dtype == "ascii":
            self._dim_tiledb_dtype = lt.DataType.STRING_ASCII
            if dim_tile is not None:
                raise TypeError(
                    "invalid tile extent, cannot set a tile a string dimension"
                )
        else:
            dim_dtype = np.dtype(dim_dtype)
            if dim_dtype.kind == "S":
                self._dim_dtype = lt.DataType.STRING_ASCII
            self._dim_dtype = dtype_to_tiledb(dim_dtype)
            if dim_tile is None:
                self._tile_extent_buffer = None
            else:
                self._tile_extent_buffer = tiledb_cast_tile_extent(dim_tile, dim_dtype)
                if self._tile_extent_buffer.size != 1:
                    raise ValueError("dimension tile extent must be a scalar")

        # Set label filters
        self._label_filters = None
        if label_filters is not None:
            if isinstance(label_filters, FilterList):
                self._label_filters = label_filters
            else:
                self._label_filters = FilterList(label_filters)

    @property
    def _dimension_tiledb_dtype(self):
        return self._dim_dtype

    @property
    def _dimension_tile_extent(self):
        return self._tile_extent_buffer

    @property
    def _label_tiledb_dtype(self):
        return self._label_dtype

    @property
    def _label_tiledb_order(self):
        return self._label_order.value

    def __repr__(self):
        # TODO Add representation
        return ""

    @property
    def dimension_index(self) -> np.uint32:
        """Index of the dimension the labels will be added to

        :rtype: numpy.uint32
        """
        return self._dim_index

    @property
    def dim_dtype(self) -> np.uint32:
        """Numpy dtype object representing the dimension type

        :rtype: numpy.dtype
        """
        return numpy_dtype(self._dim_dtype)

    @property
    def label_filters(self) -> FilterList:
        """FilterList of the labels

        :rtype: tiledb.FilterList
        :raises: :py:exc:`tiledb.TileDBError`
        """
        return (
            None
            if self._label_filters is None
            else FilterList(ctx=self._ctx, _lt_obj=self._label_filters)
        )

    @property
    def label_dtype(self) -> np.dtype:
        """Numpy dtype object representing the label type

        :rtype: numpy.dtype
        """
        return numpy_dtype(self._label_dtype)

    @property
    def label_order(self) -> str:
        """Sort order of the labels on the dimension label

        :rtype: str
        """
        return self._label_order.name
