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


def _tiledb_type_is_integer(tdb_type: lt.DataType):
    return tdb_type in (
        lt.DataType.UINT8,
        lt.DataType.INT8,
        lt.DataType.UINT16,
        lt.DataType.INT16,
        lt.DataType.UINT32,
        lt.DataType.INT32,
        lt.DataType.UINT64,
        lt.DataType.INT64,
    )


def _tiledb_type_is_datetime(tdb_type: lt.DataType):
    return tdb_type in (
        lt.DataType.DATETIME_YEAR,
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


def _tiledb_layout_string(order):
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


def _tiledb_layout(order):
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
