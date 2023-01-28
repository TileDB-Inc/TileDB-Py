from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import tiledb.cc as lt


@dataclass(frozen=True)
class DataType:
    np_dtype: np.dtype
    tiledb_type: lt.DataType

    @classmethod
    def from_numpy(cls, dtype: np.dtype) -> DataType:
        try:
            return cls(dtype, _NUMPY_TO_TILEDB[dtype])
        except KeyError:
            raise TypeError(f"{dtype!r} cannot be mapped to a DataType")


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
