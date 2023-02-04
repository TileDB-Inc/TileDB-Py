from typing import Any

import numpy as np

import tiledb.cc as lt

from .datatypes import DataType


def tiledb_type_is_datetime(tiledb_type):
    """Returns True if the tiledb type is a datetime type"""
    return np.issubdtype(DataType.from_tiledb(tiledb_type).np_dtype, np.datetime64)


def dtype_to_tiledb(dtype: np.dtype) -> lt.DataType:
    return DataType.from_numpy(dtype).tiledb_type


def array_type_ncells(dtype: np.dtype) -> lt.DataType:
    """
    Returns the TILEDB_{TYPE} and ncells corresponding to a given numpy dtype
    """
    dt = DataType.from_numpy(dtype)
    return dt.tiledb_type, dt.ncells


def dtype_range(dtype: np.dtype):
    """Return the range of a Numpy dtype"""
    return DataType.from_numpy(dtype).domain


def tiledb_cast_tile_extent(tile_extent: Any, dtype: np.dtype) -> np.array:
    """Given a tile extent value, cast it to np.array of the given numpy dtype."""
    return DataType.from_numpy(dtype).cast_tile_extent(tile_extent)


def numpy_dtype(tiledb_dtype: lt.DataType, cell_size: int = 1) -> np.dtype:
    """Return a numpy type given a tiledb_datatype_t enum value."""
    return DataType.from_tiledb(tiledb_dtype, cell_size).np_dtype
