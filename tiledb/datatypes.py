from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

import numpy as np

import tiledb.cc as lt


@dataclass(frozen=True)
class DataType:
    np_dtype: np.dtype
    tiledb_type: lt.DataType
    ncells: int
    min: Any
    max: Any

    @classmethod
    def from_numpy(cls, dtype: np.dtype) -> DataType:
        base_dtype = dtype = np.dtype(dtype)
        ncells = 1

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

        elif np.issubdtype(dtype, np.complexfloating):
            ncells = 2

        try:
            tiledb_type = _NUMPY_TO_TILEDB[base_dtype]
        except KeyError:
            raise TypeError(f"{dtype!r} cannot be mapped to a DataType")

        dtype_min, dtype_max = cls._get_min_max(base_dtype)
        return cls(dtype, tiledb_type, ncells, dtype_min, dtype_max)

    @staticmethod
    def _get_min_max(dtype: np.dtype) -> Tuple[Any, Any]:
        if dtype.kind in ("M", "m"):
            # datetime or timedelta
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
    (np.dtype("S"), lt.DataType.STRING_ASCII),
    (np.dtype("<U1"), lt.DataType.STRING_UTF8),
]
assert len(set(x for x, y in _COMMON_DATATYPES)) == len(_COMMON_DATATYPES)
assert len(set(y for x, y in _COMMON_DATATYPES)) == len(_COMMON_DATATYPES)

# numpy has complex, tiledb doesn't
_NUMPY_TO_TILEDB = {n: t for n, t in _COMMON_DATATYPES}
_NUMPY_TO_TILEDB[np.dtype("complex64")] = lt.DataType.FLOAT32
_NUMPY_TO_TILEDB[np.dtype("complex128")] = lt.DataType.FLOAT64
