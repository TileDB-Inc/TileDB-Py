from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

import numpy as np

import tiledb.cc as lt


@dataclass(frozen=True)
class DataType:
    np_dtype: np.dtype
    tiledb_type: lt.DataType
    min: Any
    max: Any

    @classmethod
    def from_numpy(cls, dtype: np.dtype) -> DataType:
        try:
            tiledb_type = _NUMPY_TO_TILEDB[dtype]
        except KeyError:
            raise ValueError(f"{dtype!r} cannot be mapped to a DataType")

        return cls(dtype, tiledb_type, *cls._get_min_max(dtype))

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

        raise ValueError(f"Cannot determine min/max for {dtype!r}")


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
