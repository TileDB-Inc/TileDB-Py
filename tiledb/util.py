import tiledb
import tiledb.cc as lt

import numpy as np
from tiledb.dataframe_ import ColumnInfo


def _sparse_schema_from_dict(input_attrs, input_dims):
    attr_infos = {k: ColumnInfo.from_values(v) for k, v in input_attrs.items()}
    dim_infos = {k: ColumnInfo.from_values(v) for k, v in input_dims.items()}

    dims = list()
    for name, dim_info in dim_infos.items():
        dim_dtype = np.bytes_ if dim_info.dtype == np.dtype("U") else dim_info.dtype
        dtype_min, dtype_max = tiledb.libtiledb.dtype_range(dim_info.dtype)

        if np.issubdtype(dim_dtype, np.integer):
            dtype_max = dtype_max - 1
        if np.issubdtype(dim_dtype, np.integer) and dtype_min < 0:
            dtype_min = dtype_min + 1

        dims.append(
            tiledb.Dim(
                name=name, domain=(dtype_min, dtype_max), dtype=dim_dtype, tile=1
            )
        )

    attrs = list()
    for name, attr_info in attr_infos.items():
        dtype_min, dtype_max = tiledb.libtiledb.dtype_range(attr_info.dtype)

        attrs.append(tiledb.Attr(name=name, dtype=dim_dtype))

    return tiledb.ArraySchema(domain=tiledb.Domain(*dims), attrs=attrs, sparse=True)


def schema_from_dict(attrs, dims):
    return _sparse_schema_from_dict(attrs, dims)


# Conversion from TileDB dtype to Numpy dtype
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

# Conversion from TileDB dtype to Numpy datetime
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


def _tiledb_type_is_datetime(tiledb_type):
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


def _tiledb_type_to_datetime(tiledb_type):
    """
    Return a datetime64 with appropriate unit for the given
    tiledb_datetype_t enum value
    """
    tdb_type = _tiledb_dtype_to_datetime_convert.get(tiledb_type, None)
    if tdb_type is None:
        raise TypeError("tiledb type is not a datetime {0!r}".format(tiledb_type))
    return tdb_type


def _numpy_dtype(tiledb_dtype, cell_size=1):
    """Return a numpy type given a tiledb_datatype_t enum value."""
    cell_val_num = cell_size

    if tiledb_dtype == lt.DataType.BLOB:
        return np.bytes_

    elif cell_val_num == 1:
        if tiledb_dtype in _tiledb_dtype_to_numpy_dtype_convert:
            return _tiledb_dtype_to_numpy_dtype_convert[tiledb_dtype]
        elif _tiledb_type_is_datetime(tiledb_dtype):
            return _tiledb_type_to_datetime(tiledb_dtype)

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
        base_dtype = _numpy_dtype(tiledb_dtype, cell_size=1)
        return base_dtype

    elif cell_val_num > 1:
        # construct anonymous record dtype
        base_dtype = _numpy_dtype(tiledb_dtype, cell_size=1)
        rec = np.dtype([("", base_dtype)] * cell_val_num)
        return rec

    raise TypeError("tiledb datatype not understood")
