from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Tuple

import numpy as np

import tiledb.cc as lt


@dataclass(frozen=True)
class DataType:
    np_dtype: np.dtype
    tiledb_type: lt.DataType
    ncells: int

    @classmethod
    @lru_cache()
    def from_numpy(cls, dtype: np.dtype) -> DataType:
        if dtype == "ascii":
            return cls(np.dtype("S"), lt.DataType.STRING_ASCII, lt.TILEDB_VAR_NUM())

        if dtype == "blob":
            return cls(np.dtype("S"), lt.DataType.BLOB, 1)

        dtype = np.dtype(dtype)
        if dtype.kind == "V":
            # fixed-size record dtypes
            if dtype.shape != ():
                raise TypeError("nested sub-array numpy dtypes are not supported")

            # check that types are the same
            field_dtypes = set(v[0] for v in dtype.fields.values())
            if len(field_dtypes) > 1:
                raise TypeError("heterogenous record numpy dtypes are not supported")

            base_dtype = field_dtypes.pop()
            ncells = len(dtype.fields)

        elif np.issubdtype(dtype, np.character):
            # - flexible datatypes of unknown size have an itemsize of 0 (str, bytes, etc.)
            # - character types are always stored as VAR because we don't want to store
            #   the pad (numpy pads to max length for 'S' and 'U' dtypes)
            base_dtype = np.dtype((dtype.kind, 1))
            if dtype.itemsize == 0:
                ncells = lt.TILEDB_VAR_NUM()
            else:
                ncells = dtype.itemsize // base_dtype.itemsize

        else:
            base_dtype = dtype
            ncells = 2 if np.issubdtype(dtype, np.complexfloating) else 1

        tiledb_type = _NUMPY_TO_TILEDB.get(base_dtype)
        if tiledb_type is None:
            raise TypeError(f"{dtype!r} cannot be mapped to a DataType")

        return cls(dtype, tiledb_type, ncells)

    @classmethod
    @lru_cache()
    def from_tiledb(cls, tiledb_type: lt.DataType, ncells: int = 1) -> DataType:
        base_dtype = _TILEDB_TO_NUMPY[tiledb_type]
        if tiledb_type in (lt.DataType.CHAR, lt.DataType.STRING_UTF8):
            kind = base_dtype.kind
            dtype = np.dtype((kind, ncells) if ncells != lt.TILEDB_VAR_NUM() else kind)
        elif ncells == 1 or ncells == lt.TILEDB_VAR_NUM():
            dtype = base_dtype
        elif ncells == 2 and np.issubdtype(base_dtype, np.floating):
            dtype = np.dtype("complex64" if base_dtype.itemsize == 4 else "complex128")
        else:
            # construct anonymous record dtype
            assert ncells > 1
            dtype = np.dtype([("", base_dtype)] * ncells)

        return cls(dtype, tiledb_type, ncells)

    @property  # TODO: change to functools.cached_property in Python 3.8+
    def domain(self) -> Tuple[Any, Any]:
        dtype = self.np_dtype

        if np.issubdtype(dtype, np.datetime64) or np.issubdtype(dtype, np.timedelta64):
            info = np.iinfo(np.int64)
            dt_data = np.datetime_data(dtype)
            # +1 to exclude NaT
            return dtype.type(info.min + 1, dt_data), dtype.type(info.max, dt_data)

        if np.issubdtype(dtype, np.integer):
            info = np.iinfo(dtype)
            return info.min, info.max

        if np.issubdtype(dtype, np.inexact):
            info = np.finfo(dtype)
            return info.min, info.max

        if np.issubdtype(dtype, np.bool_):
            return False, True

        if np.issubdtype(dtype, np.character):
            return None, None

        raise TypeError(f"Cannot determine min/max for {dtype!r}")

    def cast_tile_extent(self, tile_extent: Any) -> np.ndarray:
        """Given a tile extent value, cast it to np.array of this datatype's np_dtype."""
        if np.issubdtype(self.np_dtype, np.datetime64):
            # Special handling for datetime domains
            if isinstance(tile_extent, np.timedelta64):
                unit = np.datetime_data(self.np_dtype)[0]
                tile_extent /= np.timedelta64(1, unit)
            tile_dtype = np.dtype(np.int64)
        else:
            tile_dtype = self.np_dtype
        tile_size_array = np.array(tile_extent, tile_dtype)
        if tile_size_array.size != 1:
            raise ValueError("tile extent must be a scalar")
        return tile_size_array

    def uncast_tile_extent(self, tile_extent: Any) -> np.generic:
        """Given a tile extent value from PyBind, cast it to appropriate output."""
        if np.issubdtype(self.np_dtype, np.character):
            return tile_extent
        if np.issubdtype(self.np_dtype, np.datetime64):
            unit = np.datetime_data(self.np_dtype)[0]
            return np.timedelta64(tile_extent, unit)
        return self.np_dtype.type(tile_extent)


# datatype pairs that have a 1-1 mapping between tiledb and numpy
_COMMON_DATATYPES = [
    (np.dtype("bool"), lt.DataType.BOOL),
    # signed int
    (np.dtype("int8"), lt.DataType.INT8),
    (np.dtype("int16"), lt.DataType.INT16),
    (np.dtype("int32"), lt.DataType.INT32),
    (np.dtype("int64"), lt.DataType.INT64),
    # unsigned int
    (np.dtype("uint8"), lt.DataType.UINT8),
    (np.dtype("uint16"), lt.DataType.UINT16),
    (np.dtype("uint32"), lt.DataType.UINT32),
    (np.dtype("uint64"), lt.DataType.UINT64),
    # float
    (np.dtype("float32"), lt.DataType.FLOAT32),
    (np.dtype("float64"), lt.DataType.FLOAT64),
    # datetime
    (np.dtype("<M8[Y]"), lt.DataType.DATETIME_YEAR),
    (np.dtype("<M8[M]"), lt.DataType.DATETIME_MONTH),
    (np.dtype("<M8[W]"), lt.DataType.DATETIME_WEEK),
    (np.dtype("<M8[D]"), lt.DataType.DATETIME_DAY),
    (np.dtype("<M8[h]"), lt.DataType.DATETIME_HR),
    (np.dtype("<M8[m]"), lt.DataType.DATETIME_MIN),
    (np.dtype("<M8[s]"), lt.DataType.DATETIME_SEC),
    (np.dtype("<M8[ms]"), lt.DataType.DATETIME_MS),
    (np.dtype("<M8[us]"), lt.DataType.DATETIME_US),
    (np.dtype("<M8[ns]"), lt.DataType.DATETIME_NS),
    (np.dtype("<M8[ps]"), lt.DataType.DATETIME_PS),
    (np.dtype("<M8[fs]"), lt.DataType.DATETIME_FS),
    (np.dtype("<M8[as]"), lt.DataType.DATETIME_AS),
    # timedelta
    (np.dtype("<m8[h]"), lt.DataType.TIME_HR),
    (np.dtype("<m8[m]"), lt.DataType.TIME_MIN),
    (np.dtype("<m8[s]"), lt.DataType.TIME_SEC),
    (np.dtype("<m8[ms]"), lt.DataType.TIME_MS),
    (np.dtype("<m8[us]"), lt.DataType.TIME_US),
    (np.dtype("<m8[ns]"), lt.DataType.TIME_NS),
    (np.dtype("<m8[ps]"), lt.DataType.TIME_PS),
    (np.dtype("<m8[fs]"), lt.DataType.TIME_FS),
    (np.dtype("<m8[as]"), lt.DataType.TIME_AS),
    # byte/string
    (np.dtype("S1"), lt.DataType.CHAR),
    (np.dtype("<U1"), lt.DataType.STRING_UTF8),
]
assert len(set(x for x, y in _COMMON_DATATYPES)) == len(_COMMON_DATATYPES)
assert len(set(y for x, y in _COMMON_DATATYPES)) == len(_COMMON_DATATYPES)

# numpy has complex, tiledb doesn't
_NUMPY_TO_TILEDB = {n: t for n, t in _COMMON_DATATYPES}
_NUMPY_TO_TILEDB[np.dtype("complex64")] = lt.DataType.FLOAT32
_NUMPY_TO_TILEDB[np.dtype("complex128")] = lt.DataType.FLOAT64

# tiledb has STRING_ASCII and BLOB, numpy doesn't
_TILEDB_TO_NUMPY = {t: n for n, t in _COMMON_DATATYPES}
_TILEDB_TO_NUMPY[lt.DataType.STRING_ASCII] = np.dtype("S")
_TILEDB_TO_NUMPY[lt.DataType.BLOB] = np.dtype("S")

# pre-populate the LRU caches with all ncell=1 datatypes
list(map(DataType.from_numpy, _NUMPY_TO_TILEDB.keys()))
assert DataType.from_numpy.cache_info().currsize == len(_NUMPY_TO_TILEDB)
list(map(DataType.from_tiledb, _TILEDB_TO_NUMPY.keys()))
assert DataType.from_tiledb.cache_info().currsize == len(_TILEDB_TO_NUMPY)
