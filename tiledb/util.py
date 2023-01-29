from typing import Any

import numpy as np

import tiledb.cc as lt

from .datatypes import DataType

_str_to_data_order = {
    "unordered": lt.DataOrder.UNORDERED_DATA,
    "increasing": lt.DataOrder.INCREASING_DATA,
    "decreasing": lt.DataOrder.DECREASING_DATA,
}

_data_order_to_str = {
    lt.DataOrder.INCREASING_DATA: "increasing",
    lt.DataOrder.DECREASING_DATA: "decreasing",
    lt.DataOrder.UNORDERED_DATA: "unordered",
}


def tiledb_type_is_datetime(tiledb_type):
    """Returns True if the tiledb type is a datetime type"""
    return np.issubdtype(DataType.from_tiledb(tiledb_type).np_dtype, np.datetime64)


def dtype_to_tiledb(dtype: np.dtype) -> lt.DataType:
    return DataType.from_numpy(dtype).tiledb_type


def str_to_tiledb_data_order(order: str) -> lt.DataOrder:
    if order not in _str_to_data_order:
        raise TypeError(f"data order {order!r} not understood")
    return _str_to_data_order[order]


def tiledb_data_order_to_str(order: lt.DataOrder) -> str:
    if order not in _data_order_to_str:
        raise TypeError("unknown data order")
    return _data_order_to_str[order]


def array_type_ncells(dtype: np.dtype) -> lt.DataType:
    """
    Returns the TILEDB_{TYPE} and ncells corresponding to a given numpy dtype
    """
    dt = DataType.from_numpy(dtype)
    return dt.tiledb_type, dt.ncells


def dtype_range(dtype: np.dtype):
    """Return the range of a Numpy dtype"""
    dt = DataType.from_numpy(dtype)
    return dt.min, dt.max


def tiledb_cast_tile_extent(tile_extent: Any, dtype: np.dtype) -> np.array:
    """Given a tile extent value, cast it to np.array of the given numpy dtype."""
    # Special handling for datetime domains
    if dtype.kind == "M":
        date_unit = np.datetime_data(dtype)[0]
        if isinstance(tile_extent, np.timedelta64):
            extent_value = int(tile_extent / np.timedelta64(1, date_unit))
            tile_size_array = np.array(np.int64(extent_value), dtype=np.int64)
        else:
            tile_size_array = np.array(tile_extent, dtype=np.int64)
    else:
        tile_size_array = np.array(tile_extent, dtype=dtype)

    if tile_size_array.size != 1:
        raise ValueError("tile extent must be a scalar")

    return tile_size_array


def numpy_dtype(tiledb_dtype: lt.DataType, cell_size: int = 1) -> np.dtype:
    """Return a numpy type given a tiledb_datatype_t enum value."""
    return DataType.from_tiledb(tiledb_dtype, cell_size).np_dtype
