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


def dtype_to_tiledb(dtype):
    if dtype.name not in _dtype_to_tiledb:
        raise TypeError(f"data type {dtype!r} not understood")
    return _dtype_to_tiledb[dtype.name]


def array_type_ncells(dtype):
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
        nfields = 1
        for (typ, _) in deq:
            nfields += 1
            if typ != typ0:
                raise TypeError("heterogenous record numpy dtypes are not supported")

        tdb_type = dtype_to_tiledb(typ0)
        ncells = len(checked_dtype.fields.values())

    else:
        # scalar cell type
        tdb_type = dtype_to_tiledb(checked_dtype)
        ncells = 1

    return tdb_type, ncells
