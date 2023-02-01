from collections import deque

import numpy as np

import tiledb.cc as lt

_dtype_to_tiledb = {
    "int32": lt.DataType.INT32,
    "int64": lt.DataType.INT64,
    "float32": lt.DataType.FLOAT32,
    "float64": lt.DataType.FLOAT64,
    "int8": lt.DataType.INT8,
    "uint8": lt.DataType.UINT8,
    "int16": lt.DataType.INT16,
    "uint16": lt.DataType.UINT16,
    "uint32": lt.DataType.UINT32,
    "uint64": lt.DataType.UINT64,
    "complex64": lt.DataType.FLOAT32,
    "complex129": lt.DataType.FLOAT64,
    "datetime64[Y]": lt.DataType.DATETIME_YEAR,
    "datetime64[M]": lt.DataType.DATETIME_MONTH,
    "datetime64[W]": lt.DataType.DATETIME_WEEK,
    "datetime64[D]": lt.DataType.DATETIME_DAY,
    "datetime64[h]": lt.DataType.DATETIME_HR,
    "datetime64[m]": lt.DataType.DATETIME_MIN,
    "datetime64[s]": lt.DataType.DATETIME_SEC,
    "datetime64[ms]": lt.DataType.DATETIME_MS,
    "datetime64[us]": lt.DataType.DATETIME_US,
    "datetime64[ns]": lt.DataType.DATETIME_NS,
    "datetime64[ps]": lt.DataType.DATETIME_PS,
    "datetime64[fs]": lt.DataType.DATETIME_FS,
    "datetime64[as]": lt.DataType.DATETIME_AS,
    "timedelta64[h]": lt.DataType.TIME_HR,
    "timedelta64[m]": lt.DataType.TIME_MIN,
    "timedelta64[s]": lt.DataType.TIME_SEC,
    "timedelta64[ms]": lt.DataType.TIME_MS,
    "timedelta64[us]": lt.DataType.TIME_US,
    "timedelta64[ns]": lt.DataType.TIME_NS,
    "timedelta64[ps]": lt.DataType.TIME_PS,
    "timedelta64[fs]": lt.DataType.TIME_FS,
    "timedelta64[as]": lt.DataType.TIME_AS,
    "bool": lt.DataType.BOOL,
}


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
    if dtype.name not in _dtype_to_tiledb:
        raise TypeError(f"data type {dtype!r} not understood")
    return _dtype_to_tiledb[dtype.name]


def array_type_ncells(dtype: np.dtype) -> lt.DataType:
    """
    Returns the TILEDB_{TYPE} and ncells corresponding to a given numpy dtype
    """
    checked_dtype = np.dtype(dtype)

    # - flexible datatypes of unknown size have an itemsize of 0 (str, bytes, etc.)
    # - unicode and string types are always stored as VAR because we don't want to
    #   store the pad (numpy pads to max length for 'S' and 'U' dtypes)

    if np.issubdtype(checked_dtype, np.bytes_):
        tdb_type = lt.DataType.CHAR
        if checked_dtype.itemsize == 0:
            ncells = lt.TILEDB_VAR_NUM()
        else:
            ncells = checked_dtype.itemsize

    elif np.issubdtype(checked_dtype, np.unicode_):
        np_unicode_size = np.dtype("U1").itemsize

        # TODO depending on np_unicode_size, tdb_type may be UTF16 or UTF32
        tdb_type = lt.DataType.STRING_UTF8

        if checked_dtype.itemsize == 0:
            ncells = lt.TILEDB_VAR_NUM()
        else:
            ncells = checked_dtype.itemsize // np_unicode_size

    elif np.issubdtype(checked_dtype, np.complexfloating):
        # handle complex dtypes
        tdb_type = dtype_to_tiledb(checked_dtype)
        ncells = 2

    elif checked_dtype.kind == "V":
        # handles n fixed-size record dtypes
        if checked_dtype.shape != ():
            raise TypeError("nested sub-array numpy dtypes are not supported")
        # check that types are the same
        # TODO: make sure this is not too slow for large record types
        deq = deque(checked_dtype.fields.values())
        typ0, _ = deq.popleft()
        for typ, _ in deq:
            if typ != typ0:
                raise TypeError("heterogenous record numpy dtypes are not supported")

        tdb_type = dtype_to_tiledb(typ0)
        ncells = len(checked_dtype.fields.values())

    else:
        # scalar cell type
        tdb_type = dtype_to_tiledb(checked_dtype)
        ncells = 1

    return tdb_type, ncells


def dtype_range(dtype: np.dtype):
    """Return the range of a Numpy dtype"""

    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        dtype_min, dtype_max = info.min, info.max
    elif np.issubdtype(dtype, np.floating):
        info = np.finfo(dtype)
        dtype_min, dtype_max = info.min, info.max
    elif dtype.kind == "M":
        info = np.iinfo(np.int64)
        date_unit = np.datetime_data(dtype)[0]
        # +1 to exclude NaT
        dtype_min = np.datetime64(info.min + 1, date_unit)
        dtype_max = np.datetime64(info.max, date_unit)
    else:
        raise TypeError("invalid Dim dtype {0!r}".format(dtype))
    return (dtype_min, dtype_max)


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
