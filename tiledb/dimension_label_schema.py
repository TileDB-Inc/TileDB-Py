import numpy as np
from typing import Any, Sequence, Tuple, TYPE_CHECKING, Union

import tiledb.cc as lt
from .ctx import default_ctx
from .np2buf import dtype_to_tiledb
from .util import dtype_range, tiledb_cast_tile_extent, tiledb_type_is_datetime

from .filter import FilterList, Filter

if TYPE_CHECKING:
    from .libtiledb import Ctx


class DimLabelSchema:
    def __init__(
        self,
        dim_index: np.uint32,
        order: str = "increasing",
        label_dtype: np.dtype = np.uint64,
        dim_dtype: np.dtype = np.uint64,
        dim_tile: Any = None,
        filters: Union[FilterList, Sequence[Filter]] = None,
    ):
        self.dim_index_ = dim_index
        self.filters_ = filters

        # Get the TileDB data order from input order string.
        str_to_order = {
            "increasing": lt.DataOrder.INCREASING_DATA,
            "decreasing": lt.DataOrder.DECREASING_DATA,
            "unordered": lt.DataOrder.UNORDERED_DATA,
        }
        self.label_order_ = str_to_order[order]

        # Convert the label np.dtype to a TileDB datatype enum.
        if label_dtype is None:
            raise TypeError("TODO: Error")
        if (isinstance(label_dtype, str) and label_dtype == "ascii") or np.dtype(
            label_dtype
        ).kind == "S":
            self.label_tiledb_dtype_ = lt.DataType.STRING_ASCII
        else:
            label_dtype = np.dtype(label_dtype)
            self.label_tiledb_dtype_ = dtype_to_tiledb(label_dtype)

        # Convert the original dimension np.dtype to a TileDB datatype enum and
        # the input tile extent to an array.
        if dim_dtype is None:
            raise TypeError("TODO: Error")
        if (isinstance(dim_dtype, str) and dim_dtype == "ascii") or np.dtype(
            dim_dtype
        ).kind == "S":
            if dim_tile is not None:
                raise TypeError(
                    "invalid tile extent, cannot set a tile a string dimension"
                )
            self.tile_extent_buffer_ = None
        else:
            dim_dtype = np.dtype(dim_dtype)
            self.dim_tiledb_dtype_ = dtype_to_tiledb(dim_dtype)
            if dim_tile is None:
                self.tile_extent_buffer_ = None
            else:
                self.tile_extent_buffer_ = tiledb_cast_tile_extent(dim_tile, dim_dtype)

    @property
    def dimension_index(self):
        return self.dim_index_

    @property
    def _dimension_tiledb_dtype(self):
        return self.dimension_tiledb_dtype

    @property
    def _dimension_tile_extent(self):
        return self.tile_extent_buffer_

    @property
    def _label_tiledb_dtype(self):
        return self.label_tiledb_dtype_

    @property
    def _label_order(self):
        return self.label_order_
