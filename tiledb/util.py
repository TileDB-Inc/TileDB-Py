from collections import deque
from typing import Any, Iterable, List, Optional, Tuple, Union

import numpy as np

import tiledb
import tiledb.cc as lt

from .dataframe_ import ColumnInfo

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


def dtype_range(dtype: np.dtype) -> Tuple[Any]:
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


def schema_from_dict(attrs: List[str], dims: List[str]) -> "tiledb.ArraySchema":
    attr_infos = {k: ColumnInfo.from_values(v) for k, v in attrs.items()}
    dim_infos = {k: ColumnInfo.from_values(v) for k, v in dims.items()}

    dims = list()
    for name, dim_info in dim_infos.items():
        dim_dtype = np.bytes_ if dim_info.dtype == np.dtype("U") else dim_info.dtype
        dtype_min, dtype_max = dtype_range(dim_info.dtype)

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
        dtype_min, dtype_max = dtype_range(attr_info.dtype)

        attrs.append(tiledb.Attr(name=name, dtype=dim_dtype))

    return tiledb.ArraySchema(domain=tiledb.Domain(*dims), attrs=attrs, sparse=True)


def tiledb_type_to_datetime(tiledb_type: lt.DataType):
    """
    Return a datetime64 with appropriate unit for the given
    tiledb_datetype_t enum value
    """
    tdb_type = _tiledb_dtype_to_datetime_convert.get(tiledb_type, None)
    if tdb_type is None:
        raise TypeError("tiledb type is not a datetime {0!r}".format(tiledb_type))
    return tdb_type


def tiledb_type_is_integer(tiledb_type: lt.DataType):
    return tiledb_type in (
        lt.DataType.UINT8,
        lt.DataType.INT8,
        lt.DataType.UINT16,
        lt.DataType.INT16,
        lt.DataType.UINT32,
        lt.DataType.INT32,
        lt.DataType.UINT64,
        lt.DataType.INT64,
    )


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


def sparse_array_from_numpy(
    uri: str, array: np.array, ctx: Optional["tiledb.Ctx"] = None, **kw
):
    """
    Implementation of tiledb.from_numpy for dense arrays. See documentation
    of tiledb.from_numpy.
    """
    if not ctx:
        ctx = tiledb.default_ctx()

    mode = kw.pop("mode", "ingest")
    timestamp = kw.pop("timestamp", None)

    if mode not in ("ingest", "schema_only", "append"):
        raise tiledb.TileDBError(f"Invalid mode specified ('{mode}')")

    if mode in ("ingest", "schema_only"):
        try:
            with tiledb.Array.load_typed(uri):
                raise tiledb.TileDBError(f"Array URI '{uri}' already exists!")
        except tiledb.TileDBError:
            pass

    if mode == "append":
        kw["append_dim"] = kw.get("append_dim", 0)
        if tiledb.ArraySchema.load(uri).sparse:
            raise tiledb.TileDBError("Cannot append to sparse array")

    if mode in ("ingest", "schema_only"):
        schema = _schema_like_numpy(array, ctx=ctx, **kw)
        tiledb.Array.create(uri, schema)

    if mode in ("ingest", "append"):
        kw["mode"] = mode
        with tiledb.open(uri, mode="w", ctx=ctx, timestamp=timestamp) as arr:
            # <TODO> probably need better typecheck here
            if array.dtype == object:
                arr[:] = array
            else:
                arr.write_direct(np.ascontiguousarray(array), **kw)

    return tiledb.DenseArray(uri, mode="r", ctx=ctx)


def schema_like(
    *args,
    shape: Optional[tuple] = None,
    dtype: Optional[np.dtype] = None,
    ctx: Optional["tiledb.Ctx"] = None,
    **kw,
) -> "tiledb.ArraySchema":
    """
    Return an ArraySchema corresponding to a NumPy-like object or
    `shape` and `dtype` kwargs. Users are encouraged to pass 'tile'
    and 'capacity' keyword arguments as appropriate for a given
    application.

    :param A: NumPy array-like object, or TileDB reference URI, optional
    :param tuple shape: array shape, optional
    :param dtype: array dtype, optional
    :param Ctx ctx: TileDB Ctx
    :param kwargs: additional keyword arguments to pass through, optional
    :return: tiledb.ArraySchema
    """
    if not ctx:
        ctx = tiledb.default_ctx()

    def is_ndarray_like(arr):
        return hasattr(arr, "shape") and hasattr(arr, "dtype") and hasattr(arr, "ndim")

    # support override of default dimension dtype
    dim_dtype = kw.pop("dim_dtype", np.uint64)
    if len(args) == 1:
        arr = args[0]
        if is_ndarray_like(arr):
            tiling = _regularize_tiling(kw.pop("tile", None), arr.ndim)
            schema = _schema_like_numpy(arr, tile=tiling, dim_dtype=dim_dtype, ctx=ctx)
        else:
            raise ValueError("expected ndarray-like object")
    elif shape and dtype:
        if np.issubdtype(np.bytes_, dtype):
            dtype = np.dtype("S")
        elif np.issubdtype(dtype, np.unicode_):
            dtype = np.dtype("U")

        ndim = len(shape)
        tiling = _regularize_tiling(kw.pop("tile", None), ndim)

        dims = []
        for d in range(ndim):
            # support smaller tile extents by kw
            # domain is based on full shape
            tile_extent = tiling[d] if tiling else shape[d]
            domain = (0, shape[d] - 1)
            dims.append(
                tiledb.Dim(domain=domain, tile=tile_extent, dtype=dim_dtype, ctx=ctx)
            )

        att = tiledb.Attr(dtype=dtype, ctx=ctx)
        dom = tiledb.Domain(*dims, ctx=ctx)
        schema = tiledb.ArraySchema(ctx=ctx, domain=dom, attrs=(att,), **kw)
    elif kw is not None:
        raise ValueError
    else:
        raise ValueError(
            "Must provide either ndarray-like object or 'shape' "
            "and 'dtype' keyword arguments"
        )

    return schema


def _schema_like_numpy(array: np.array, ctx: Optional["tiledb.Ctx"] = None, **kw):
    """
    Internal helper function for schema_like to create array schema from
    NumPy array-like object.
    """
    if not ctx:
        ctx = tiledb.default_ctx()
    # create an ArraySchema from the numpy array object
    tiling = _regularize_tiling(kw.pop("tile", None), array.ndim)

    attr_name = kw.pop("attr_name", "")
    dim_dtype = kw.pop("dim_dtype", np.dtype("uint64"))
    full_domain = kw.pop("full_domain", False)
    dims = []

    for d in range(array.ndim):
        # support smaller tile extents by kw
        # domain is based on full shape
        tile_extent = tiling[d] if tiling else array.shape[d]
        if full_domain:
            if dim_dtype not in (np.bytes_, np.str_):
                # Use the full type domain, deferring to the constructor
                dtype_min, dtype_max = dtype_range(dim_dtype)
                dim_max = dtype_max
                if dim_dtype.kind == "M":
                    date_unit = np.datetime_data(dim_dtype)[0]
                    dim_min = np.datetime64(dtype_min, date_unit)
                    tile_max = np.iinfo(np.uint64).max - tile_extent
                    if np.uint64(dtype_max - dtype_min) > tile_max:
                        dim_max = np.datetime64(dtype_max - tile_extent, date_unit)
                else:
                    dim_min = dtype_min

                if np.issubdtype(dim_dtype, np.integer):
                    tile_max = np.iinfo(np.uint64).max - tile_extent
                    if np.uint64(dtype_max - dtype_min) > tile_max:
                        dim_max = dtype_max - tile_extent
                domain = (dim_min, dim_max)
            else:
                domain = (None, None)

            if np.issubdtype(dim_dtype, np.integer) or dim_dtype.kind == "M":
                # we can't make a tile larger than the dimension range or lower than 1
                tile_extent = max(1, min(tile_extent, np.uint64(dim_max - dim_min)))
            elif np.issubdim_dtype(dim_dtype, np.floating):
                # this difference can be inf
                with np.errstate(over="ignore"):
                    dim_range = dim_max - dim_min
                if dim_range < tile_extent:
                    tile_extent = np.ceil(dim_range)
        else:
            domain = (0, array.shape[d] - 1)

        dims.append(
            tiledb.Dim(domain=domain, tile=tile_extent, dtype=dim_dtype, ctx=ctx)
        )

    var = False
    if array.dtype == object:
        # for object arrays, we use the dtype of the first element
        # consistency check should be done later, if needed
        el0 = array.flat[0]
        if type(el0) is bytes:
            el_dtype = np.dtype("S")
            var = True
        elif type(el0) is str:
            el_dtype = np.dtype("U")
            var = True
        elif type(el0) == np.ndarray:
            if len(el0.shape) != 1:
                raise TypeError(
                    "Unsupported sub-array type for Attribute: {} "
                    "(only string arrays and 1D homogeneous NumPy arrays are supported)".format(
                        type(el0)
                    )
                )
            el_dtype = el0.dtype
        else:
            raise TypeError(
                "Unsupported sub-array type for Attribute: {} "
                "(only strings and homogeneous-typed NumPy arrays are supported)".format(
                    type(el0)
                )
            )
    else:
        el_dtype = array.dtype

    att = tiledb.Attr(dtype=el_dtype, name=attr_name, var=var, ctx=ctx)
    dom = tiledb.Domain(*dims, ctx=ctx)
    return tiledb.ArraySchema(ctx=ctx, domain=dom, attrs=(att,), **kw)


def _regularize_tiling(tile: Union[Iterable, np.isscalar], ndim: int) -> Tuple[Any]:
    """
    Internal helper function for schema_like and schema_like_numpy to regularize tiling.
    """
    if not tile:
        return None

    if np.isscalar(tile):
        return tuple(int(tile) for _ in range(ndim))

    if isinstance(tile, str) or len(tile) != ndim:
        raise ValueError("'tile' must be iterable and match array dimensionality")

    return tuple(tile)


def tiledb_layout_string(order):
    tiledb_order_to_string = {
        lt.LayoutType.ROW_MAJOR: "row-major",
        lt.LayoutType.COL_MAJOR: "col-major",
        lt.LayoutType.GLOBAL_ORDER: "global",
        lt.LayoutType.UNORDERED: "unordered",
        lt.LayoutType.HILBERT: "hilbert",
    }

    if order not in tiledb_order_to_string:
        raise ValueError(f"unknown tiledb layout: {order}")

    return tiledb_order_to_string[order]


def tiledb_layout(order):
    string_to_tiledb_order = {
        "row-major": lt.LayoutType.ROW_MAJOR,
        "C": lt.LayoutType.ROW_MAJOR,
        "col-major": lt.LayoutType.COL_MAJOR,
        "R": lt.LayoutType.COL_MAJOR,
        "global": lt.LayoutType.GLOBAL_ORDER,
        "hilbert": lt.LayoutType.HILBERT,
        "H": lt.LayoutType.HILBERT,
        "unordered": lt.LayoutType.UNORDERED,
        "U": lt.LayoutType.UNORDERED,
        None: lt.LayoutType.UNORDERED,
    }

    if order not in string_to_tiledb_order:
        raise ValueError(f"unknown tiledb layout: {order}")

    return string_to_tiledb_order[order]
