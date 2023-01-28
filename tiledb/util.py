from typing import Any

import numpy as np

import tiledb.cc as lt

from .datatypes import DataType

_tiledb_dtype_to_numpy_dtype_convert = {
    lt.DataType.INT32: np.int32,
    lt.DataType.UINT32: np.uint32,
    lt.DataType.INT64: np.int64,
    lt.DataType.UINT64: np.uint64,
    lt.DataType.FLOAT32: np.float32,
    lt.DataType.FLOAT64: np.float64,
    lt.DataType.INT8: np.int8,
    lt.DataType.UINT8: np.uint8,
    lt.DataType.INT16: np.int16,
    lt.DataType.UINT16: np.uint16,
    lt.DataType.CHAR: np.dtype("S1"),
    lt.DataType.STRING_ASCII: np.dtype("S"),
    lt.DataType.STRING_UTF8: np.dtype("U1"),
    lt.DataType.BLOB: np.byte,
    lt.DataType.BOOL: np.bool_,
}

_tiledb_dtype_to_datetime_convert = {
    lt.DataType.DATETIME_YEAR: np.datetime64("", "Y"),
    lt.DataType.DATETIME_MONTH: np.datetime64("", "M"),
    lt.DataType.DATETIME_WEEK: np.datetime64("", "W"),
    lt.DataType.DATETIME_DAY: np.datetime64("", "D"),
    lt.DataType.DATETIME_HR: np.datetime64("", "h"),
    lt.DataType.DATETIME_MIN: np.datetime64("", "m"),
    lt.DataType.DATETIME_SEC: np.datetime64("", "s"),
    lt.DataType.DATETIME_MS: np.datetime64("", "ms"),
    lt.DataType.DATETIME_US: np.datetime64("", "us"),
    lt.DataType.DATETIME_NS: np.datetime64("", "ns"),
    lt.DataType.DATETIME_PS: np.datetime64("", "ps"),
    lt.DataType.DATETIME_FS: np.datetime64("", "fs"),
    lt.DataType.DATETIME_AS: np.datetime64("", "as"),
}

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
    return tiledb_type in (
        lt.DataType.DATETIME_YEAR,
        lt.DataType.DATETIME_MONTH,
        lt.DataType.DATETIME_WEEK,
        lt.DataType.DATETIME_DAY,
        lt.DataType.DATETIME_HR,
        lt.DataType.DATETIME_MIN,
        lt.DataType.DATETIME_SEC,
        lt.DataType.DATETIME_MS,
        lt.DataType.DATETIME_US,
        lt.DataType.DATETIME_NS,
        lt.DataType.DATETIME_PS,
        lt.DataType.DATETIME_FS,
        lt.DataType.DATETIME_AS,
    )


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


def tiledb_type_to_datetime(tiledb_type: lt.DataType):
    """
    Return a datetime64 with appropriate unit for the given
    tiledb_datetype_t enum value
    """
    tdb_type = _tiledb_dtype_to_datetime_convert.get(tiledb_type, None)
    if tdb_type is None:
        raise TypeError("tiledb type is not a datetime {0!r}".format(tiledb_type))
    return tdb_type


def numpy_dtype(tiledb_dtype: lt.DataType, cell_size: int = 1) -> np.dtype:
    """Return a numpy type given a tiledb_datatype_t enum value."""
    cell_val_num = cell_size

    if tiledb_dtype == lt.DataType.BLOB:
        return np.bytes_

    elif cell_val_num == 1:
        if tiledb_dtype in _tiledb_dtype_to_numpy_dtype_convert:
            return _tiledb_dtype_to_numpy_dtype_convert[tiledb_dtype]
        elif tiledb_type_is_datetime(tiledb_dtype):
            return tiledb_type_to_datetime(tiledb_dtype)

    elif cell_val_num == 2 and tiledb_dtype == lt.DataType.FLOAT32:
        return np.complex64

    elif cell_val_num == 2 and tiledb_dtype == lt.DataType.FLOAT64:
        return np.complex128

    elif tiledb_dtype in (lt.DataType.CHAR, lt.DataType.STRING_UTF8):
        if tiledb_dtype == lt.DataType.CHAR:
            dtype_str = "|S"
        elif tiledb_dtype == lt.DataType.STRING_UTF8:
            dtype_str = "|U"
        if cell_val_num != lt.TILEDB_VAR_NUM():
            dtype_str += str(cell_val_num)
        return np.dtype(dtype_str)

    elif cell_val_num == lt.TILEDB_VAR_NUM():
        base_dtype = numpy_dtype(tiledb_dtype, cell_size=1)

        return base_dtype

    elif cell_val_num > 1:
        # construct anonymous record dtype
        base_dtype = numpy_dtype(tiledb_dtype, cell_size=1)
        return np.dtype([("", base_dtype)] * cell_val_num)

    raise TypeError("tiledb datatype not understood")
