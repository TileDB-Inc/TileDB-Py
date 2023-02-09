from enum import Enum
from typing import Any, Sequence, Union

import numpy as np

import tiledb.cc as lt

from .ctx import Ctx, default_ctx
from .datatypes import DataType
from .filter import Filter, FilterList


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
        self._ctx = ctx or default_ctx()
        self._dim_index = dim_index
        self._label_order = DataOrder[order]
        self._label_dtype = DataType.from_numpy(label_dtype)
        self._dim_dtype = DataType.from_numpy(dim_dtype)

        if dim_tile is not None:
            if np.issubdtype(self._dim_dtype.np_dtype, np.bytes_):
                raise TypeError(
                    "invalid tile extent, cannot set a tile a string dimension"
                )
            dim_tile = self._dim_dtype.cast_tile_extent(dim_tile)
        self._dim_tile = dim_tile

        if label_filters is None or isinstance(label_filters, FilterList):
            self._label_filters = label_filters
        else:
            self._label_filters = FilterList(label_filters)

    @property
    def _dimension_tiledb_dtype(self) -> lt.DataType:
        return self._dim_dtype.tiledb_type

    @property
    def _label_tiledb_dtype(self) -> lt.DataType:
        return self._label_dtype.tiledb_type

    @property
    def _label_tiledb_order(self) -> lt.DataOrder:
        return self._label_order.value

    @property
    def dimension_index(self) -> np.uint32:
        """Index of the dimension the labels will be added to"""
        return self._dim_index

    @property
    def dim_dtype(self) -> np.dtype:
        """Numpy dtype object representing the dimension type"""
        return self._dim_dtype.np_dtype

    @property
    def label_filters(self) -> FilterList:
        """FilterList of the labels"""
        return self._label_filters

    @property
    def label_dtype(self) -> np.dtype:
        """Numpy dtype object representing the label type"""
        return self._label_dtype.np_dtype

    @property
    def label_order(self) -> str:
        """Sort order of the labels on the dimension label"""
        return self._label_order.name
