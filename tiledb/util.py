import tiledb
import numpy as np
from typing import Iterable
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
