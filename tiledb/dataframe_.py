import copy
import json
import os
import warnings
from dataclasses import dataclass
from typing import Any, List, Optional, Union

import numpy as np

import tiledb

from .datatypes import DataType


def check_dataframe_deps():
    pd_error = """Pandas version >= 1.0 and < 3.0 required for dataframe functionality.
                  Please `pip install pandas>=1.0,<3.0` to proceed."""
    pa_error = """PyArrow version >= 1.0 is suggested for dataframe functionality.
                  Please `pip install pyarrow>=1.0`."""

    try:
        import pandas as pd
    except ImportError:
        raise Exception(pd_error)

    from packaging.version import Version

    if Version(pd.__version__) < Version("1.0") or Version(pd.__version__) >= Version(
        "3.0.0.dev0"
    ):
        raise Exception(pd_error)

    try:
        import pyarrow as pa

        if Version(pa.__version__) < Version("1.0"):
            warnings.warn(pa_error)
    except ImportError:
        warnings.warn(pa_error)


# Note: 'None' is used to indicate optionality for many of these options
#       For example, if the `sparse` argument is unspecified we will default
#       to False (dense) unless the input has string or heterogenous indexes.
TILEDB_KWARG_DEFAULTS = {
    "ctx": None,
    "sparse": None,
    "index_dims": None,
    "allows_duplicates": True,
    "mode": "ingest",
    "attr_filters": True,
    "dim_filters": True,
    "coords_filters": True,
    "offsets_filters": True,
    "full_domain": False,
    "tile": None,
    "row_start_idx": None,
    "fillna": None,
    "column_types": None,
    "varlen_types": None,
    "capacity": None,
    "date_spec": None,
    "cell_order": "row-major",
    "tile_order": "row-major",
    "timestamp": None,
    "debug": None,
}


def parse_tiledb_kwargs(kwargs):
    parsed_args = dict(TILEDB_KWARG_DEFAULTS)

    for key in TILEDB_KWARG_DEFAULTS.keys():
        if key in kwargs:
            parsed_args[key] = kwargs.pop(key)

    return parsed_args


def _infer_dtype_from_pandas(values, column_name):
    from pandas.api import types as pd_types

    inferred_dtype = pd_types.infer_dtype(values)
    if inferred_dtype == "bytes":
        return np.bytes_
    elif inferred_dtype == "string":
        return "<U0"
    elif inferred_dtype == "floating":
        return np.float64
    elif inferred_dtype == "integer":
        return np.int64
    elif inferred_dtype == "mixed-integer":
        raise NotImplementedError(
            f"Pandas type 'mixed-integer' is not supported (column {column_name})"
        )
    elif inferred_dtype == "mixed-integer-float":
        raise NotImplementedError(
            f"Pandas type 'mixed-integer-float' is not supported (column {column_name})"
        )
    elif inferred_dtype == "decimal":
        return np.float64
    elif inferred_dtype == "complex":
        return np.complex128
    elif inferred_dtype == "categorical":
        raise NotImplementedError(
            f"Pandas type 'categorical of categorical' is not supported (column {column_name})"
        )
    elif inferred_dtype == "boolean":
        return np.bool_
    elif inferred_dtype == "datetime64":
        return np.datetime64
    elif inferred_dtype == "datetime":
        return np.datetime64
    elif inferred_dtype == "date":
        return np.datetime64
    elif inferred_dtype == "timedelta64":
        return np.timedelta64
    elif inferred_dtype == "timedelta":
        return np.timedelta64
    elif inferred_dtype == "time":
        return np.timedelta64
    elif inferred_dtype == "period":
        raise NotImplementedError(
            f"Pandas type 'period' is not supported (column {column_name})"
        )
    elif inferred_dtype == "mixed":
        raise NotImplementedError(
            f"Pandas type 'mixed' is not supported (column {column_name})"
        )
    elif inferred_dtype == "unknown-array":
        raise NotImplementedError(
            f"Pandas type 'unknown-array' is not supported (column {column_name})"
        )


@dataclass(frozen=True)
class EnumerationInfo:
    dtype: np.dtype
    ordered: bool = False
    values: List[Any] = None


@dataclass(frozen=True)
class ColumnInfo:
    dtype: np.dtype
    repr: Optional[str] = None
    nullable: bool = False
    var: bool = False
    enumeration: bool = False
    enumeration_info: Optional[EnumerationInfo] = None

    @classmethod
    def from_values(cls, array_like, varlen_types=()):
        from pandas import CategoricalDtype
        from pandas.api import types as pd_types

        if pd_types.is_object_dtype(array_like):
            # Note: this does a full scan of the column... not sure what else to do here
            #       because Pandas allows mixed string column types (and actually has
            #       problems w/ allowing non-string types in object columns)
            inferred_dtype = pd_types.infer_dtype(array_like)
            if inferred_dtype == "bytes":
                return cls.from_dtype(np.bytes_, array_like.name)
            elif inferred_dtype == "string":
                # TODO we need to make sure this is actually convertible
                return cls.from_dtype(np.str_, array_like.name)
            else:
                raise NotImplementedError(
                    f"{inferred_dtype} inferred dtype not supported (column {array_like.name})"
                )
        elif hasattr(array_like, "dtype") and isinstance(
            array_like.dtype, CategoricalDtype
        ):
            return cls.from_categorical(
                array_like.cat, array_like.dtype, array_like.name
            )
        else:
            if not hasattr(array_like, "dtype"):
                array_like = np.asanyarray(array_like)
            return cls.from_dtype(array_like.dtype, array_like.name, varlen_types)

    @classmethod
    def from_categorical(cls, cat, dtype, column_name):
        values = cat.categories.values
        inferred_dtype = _infer_dtype_from_pandas(values, column_name)

        return cls(
            np.int32,
            repr=dtype.name,
            nullable=False,
            var=False,
            enumeration=True,
            enumeration_info=EnumerationInfo(
                values=values, ordered=cat.ordered, dtype=inferred_dtype
            ),
        )

    @classmethod
    def from_dtype(cls, dtype, column_name, varlen_types=()):
        from pandas.api import types as pd_types

        if isinstance(dtype, str) and dtype == "ascii":
            return cls("ascii", var=True)

        dtype = pd_types.pandas_dtype(dtype)
        # Note: be careful if you rearrange the order of the following checks

        # extension types
        if pd_types.is_extension_array_dtype(dtype):
            if tiledb.libtiledb.version() < (2, 10) and pd_types.is_bool_dtype(dtype):
                # as of libtiledb 2.10, we support TILEDB_BOOL
                np_type = np.uint8
            else:
                # XXX Parametrized dtypes such as "foo[int32]") sometimes have a "subtype"
                # property that holds the "int32". If it exists use this, otherwise use
                # the standard type property
                np_type = getattr(dtype, "subtype", dtype.type)

            var = bool(varlen_types and dtype in varlen_types)
            if var:
                # currently TileDB-py doesn't support nullable var-length attributes
                nullable = False
            else:
                # currently nullability is a (private) property of ExtensionArray
                # see https://github.com/pandas-dev/pandas/issues/40574
                nullable = bool(dtype.construct_array_type()._can_hold_na)

            return cls(np.dtype(np_type), repr=dtype.name, nullable=nullable, var=var)

        # bool type
        if pd_types.is_bool_dtype(dtype):
            # as of libtiledb 2.10, we support TILEDB_BOOL
            dtype = "uint8" if tiledb.libtiledb.version() < (2, 10) else "bool"
            return cls(np.dtype(dtype), repr="bool")

        # complex types
        if pd_types.is_complex_dtype(dtype):
            raise NotImplementedError(
                f"complex dtype not supported (column {column_name})"
            )

        # remaining numeric types
        if pd_types.is_numeric_dtype(dtype):
            if dtype == np.float16 or hasattr(np, "float128") and dtype == np.float128:
                raise NotImplementedError(
                    f"Only single and double precision float dtypes are supported (column {column_name})"
                )
            return cls(dtype)

        # datetime types
        if pd_types.is_datetime64_any_dtype(dtype):
            if dtype == "datetime64[ns]":
                return cls(dtype)
            else:
                raise NotImplementedError(
                    f"Only 'datetime64[ns]' datetime dtype is supported (column {column_name})"
                )

        # string types
        # don't use pd_types.is_string_dtype() because it includes object types too
        if dtype.type in (np.bytes_, np.str_):
            # str and bytes are always stored as var-length
            return cls(dtype, var=True)

        raise NotImplementedError(f"{dtype} dtype not supported (column {column_name})")


def _get_column_infos(df, column_types, varlen_types):
    column_infos = {}
    for name, column in df.items():
        if column_types and name in column_types:
            column_infos[name] = ColumnInfo.from_dtype(
                column_types[name], name, varlen_types
            )
        else:
            column_infos[name] = ColumnInfo.from_values(column, varlen_types)
    return column_infos


def _get_schema_filters(filters):
    if filters is True:
        # default case, use ZstdFilter
        return tiledb.FilterList([tiledb.ZstdFilter()])
    elif filters is None:
        # empty filter list (schema uses zstd by default if unspecified)
        return tiledb.FilterList()
    elif isinstance(filters, (list, tiledb.FilterList)):
        return tiledb.FilterList(filters)
    elif isinstance(filters, tiledb.Filter):
        return tiledb.FilterList([filters])
    else:
        raise ValueError("Unknown FilterList type!")


def _get_attr_dim_filters(name, filters):
    if isinstance(filters, dict):
        # support passing a dict of filters per-attribute
        return _get_schema_filters(filters.get(name, True))
    else:
        return _get_schema_filters(filters)


def _get_enums(names, column_infos):
    enums = []
    for name in names:
        column_info = column_infos[name]
        if not column_info.enumeration:
            continue
        enums.append(
            tiledb.Enumeration(
                name=name,
                # Pandas categoricals are always ordered
                ordered=column_info.enumeration_info.ordered,
                values=np.array(
                    column_info.enumeration_info.values,
                    dtype=column_info.enumeration_info.dtype,
                ),
            )
        )

    return enums


def _get_attrs(names, column_infos, attr_filters):
    attrs = []
    attr_reprs = {}
    for name in names:
        filters = _get_attr_dim_filters(name, attr_filters)
        column_info = column_infos[name]
        if column_info.enumeration:
            attrs.append(
                tiledb.Attr(
                    name=name,
                    filters=filters,
                    dtype=np.int32,
                    enum_label=name,
                    nullable=column_info.nullable,
                    var=column_info.var,
                )
            )
        else:
            attrs.append(
                tiledb.Attr(
                    name=name,
                    filters=filters,
                    dtype=column_info.dtype,
                    nullable=column_info.nullable,
                    var=column_info.var,
                )
            )

        if column_info.repr is not None:
            attr_reprs[name] = column_info.repr

    return attrs, attr_reprs


def create_dim(dtype, values, full_domain, tile, **kwargs):
    if not full_domain:
        values = np.asarray(values)
        dim_min = values.min()
        dim_max = values.max()
    elif not np.issubdtype(dtype, np.character):
        dim_min, dim_max = dtype_min, dtype_max = DataType.from_numpy(dtype).domain
        # TODO: simplify this logic and/or move to DataType.domain
        if np.issubdtype(dtype, np.datetime64):
            date_unit = np.datetime_data(dtype)[0]
            dim_min = np.datetime64(dtype_min, date_unit)
            tile_max = np.iinfo(np.uint64).max - tile
            if np.uint64(dtype_max - dtype_min) > tile_max:
                dim_max = np.datetime64(dtype_max - tile, date_unit)
        elif np.issubdtype(dtype, np.integer):
            tile_max = np.iinfo(np.uint64).max - tile
            if np.uint64(dtype_max - dtype_min) > tile_max:
                dim_max = dtype_max - tile
    else:
        dim_min, dim_max = None, None

    # TODO: simplify this logic and/or move to DataType.cast_tile_extent
    if np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, np.datetime64):
        # we can't make a tile larger than the dimension range or lower than 1
        tile = max(1, min(tile, 1 + np.uint64(dim_max - dim_min)))
    elif np.issubdtype(dtype, np.floating):
        # this difference can be inf
        with np.errstate(over="ignore"):
            dim_range = dim_max - dim_min
        if dim_range < tile:
            tile = np.ceil(dim_range)

    return tiledb.Dim(
        domain=(dim_min, dim_max),
        # libtiledb only supports TILEDB_ASCII dimensions, so we must use
        # nb.bytes_ which will force encoding on write
        dtype=np.bytes_ if dtype == np.str_ else dtype,
        tile=tile,
        **kwargs,
    )


def _sparse_from_dtypes(dtypes, sparse=None):
    if any(dtype in (np.bytes_, np.str_) for dtype in dtypes):
        if sparse is False:
            raise tiledb.TileDBError(
                "Cannot create dense array with string-typed dimensions"
            )
        if sparse is None:
            return True

    dtype0 = next(iter(dtypes))
    if not all(dtype0 == dtype for dtype in dtypes):
        if sparse is False:
            raise tiledb.TileDBError(
                "Cannot create dense array with heterogeneous dimension data types"
            )
        if sparse is None:
            return True

    # Fall back to default dense type if unspecified and not inferred from dimension types
    return sparse if sparse is not None else False


def create_dims(df, index_dims, tile=None, full_domain=False, filters=None):
    check_dataframe_deps()
    import pandas as pd

    per_dim_tile = isinstance(tile, dict)
    if tile is not None:
        tile_values = tile.values() if per_dim_tile else (tile,)
        if not all(isinstance(v, (int, float)) for v in tile_values):
            raise ValueError(
                "Invalid tile kwarg: expected int or dict of column names mapped to ints. "
                f"Got '{tile!r}'"
            )

    index = df.index
    name_dtype_values = []
    dim_metadata = {}

    for name in index_dims or index.names:
        if name in index.names:
            values = index.get_level_values(name)
        elif name in df.columns:
            values = df[name]
        else:
            raise ValueError(f"Unknown column or index named {name!r}")

        dtype = ColumnInfo.from_values(values).dtype
        internal_dtype = dtype

        if name == "__tiledb_rows" and isinstance(index, pd.RangeIndex):
            internal_dtype = np.dtype("uint64")
        if name is None:
            name = "__tiledb_rows"

        dim_metadata[name] = dtype
        name_dtype_values.append((name, internal_dtype, values))

    ndim = len(name_dtype_values)
    default_dim_tile = (
        10000 if ndim == 1 else 1000 if ndim == 2 else 100 if ndim == 3 else 10
    )

    def get_dim_tile(name):
        dim_tile = tile.get(name) if per_dim_tile else tile
        return dim_tile if dim_tile is not None else default_dim_tile

    dims = [
        create_dim(
            dtype=dtype,
            values=values,
            full_domain=full_domain,
            tile=get_dim_tile(name),
            name=name,
            filters=_get_attr_dim_filters(name, filters),
        )
        for name, dtype, values in name_dtype_values
    ]

    return dims, dim_metadata


def write_array_metadata(array, attr_metadata=None, index_metadata=None):
    """
    :param array: open, writable TileDB array
    :param metadata: dict
    :return:
    """
    if attr_metadata:
        attr_md_dict = {n: str(t) for n, t in attr_metadata.items()}
        array.meta["__pandas_attribute_repr"] = json.dumps(attr_md_dict)
    if index_metadata:
        index_md_dict = {n: str(t) for n, t in index_metadata.items()}
        array.meta["__pandas_index_dims"] = json.dumps(index_md_dict)


def _df_to_np_arrays(df, column_infos, fillna):
    ret = {}
    nullmaps = {}
    for name, column in df.items():
        column_info = column_infos[name]
        if fillna is not None and name in fillna:
            column = column.fillna(fillna[name])

        to_numpy_kwargs = {}

        if not column_info.var:
            to_numpy_kwargs.update(dtype=column_info.dtype)

        if column_info.nullable:
            # use default 0/empty for the dtype
            to_numpy_kwargs.update(na_value=column_info.dtype.type())
            nullmaps[name] = (~column.isna()).to_numpy(dtype=np.uint8)

        if column_info.enumeration:
            # Enumerations should get the numerical codes instead of converting enumeration values
            ret[name] = column.cat.codes.to_numpy(**to_numpy_kwargs)
        else:
            ret[name] = column.to_numpy(**to_numpy_kwargs)

    return ret, nullmaps


def from_pandas(uri, dataframe, **kwargs):
    """Create TileDB array at given URI from a Pandas dataframe

    Supports most Pandas series types, including nullable integers and
    bools.

    :param uri: URI for new TileDB array
    :param dataframe: pandas DataFrame

    :Keyword Arguments:

        * Any `pandas.read_csv <https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html>`_ supported keyword argument
        * **ctx** - A TileDB context
        * **sparse** - (default True) Create sparse schema
        * **chunksize** - (default None) Maximum number of rows to read at a time. Note that this is also a `pandas.read_csv` argument
                          which `tiledb.read_csv` checks for in order to correctly read a file batchwise.
        * **index_dims** (``List[str]``) -- List of column name(s) to use as dimension(s) in TileDB array schema. This is the recommended way to create dimensions.
        * **allows_duplicates** - Generated schema should allow duplicates
        * **mode** - Creation mode, one of 'ingest' (default), 'schema_only', 'append'
        * **attr_filters** - FilterList to apply to Attributes: FilterList or Dict[str -> FilterList] for any attribute(s). Unspecified attributes will use default.
        * **dim_filters** - FilterList to apply to Dimensions: FilterList or Dict[str -> FilterList] for any dimensions(s). Unspecified dimensions will use default.
        * **offsets_filters** - FilterList to apply to all offsets
        * **full_domain** - Dimensions should be created with full range of the dtype
        * **tile** - Dimension tiling: accepts either an int that applies the tiling to all dimensions or a dict("dim_name": int) to specifically assign tiling to a given dimension
        * **row_start_idx** - Start index to start new write (for row-indexed ingestions).
        * **fillna** - Value to use to fill holes
        * **column_types** - Dictionary of {``column_name``: dtype} to apply dtypes to columns
        * **varlen_types** - A set of {dtypes}; any column wihin the set is converted to a variable length attribute
        * **capacity** - Schema capacity.
        * **date_spec** - Dictionary of {``column_name``: format_spec} to apply to date/time columns which are not correctly inferred by pandas 'parse_dates'. Format must be specified using the Python format codes: https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior
        * **cell_order** - (default 'row-major) Schema cell order: 'row-major', 'col-major', or 'hilbert'
        * **tile_order** - (default 'row-major) Schema tile order: 'row-major' or 'col-major'
        * **timestamp** - Write TileDB array at specific timestamp.

    :raises: :py:exc:`tiledb.TileDBError`
    :return: None

    """
    check_dataframe_deps()

    if "tiledb_args" in kwargs:
        tiledb_args = kwargs.pop("tiledb_args")
    else:
        tiledb_args = parse_tiledb_kwargs(kwargs)

    ctx = tiledb_args.get("ctx")
    if ctx is not None and not isinstance(ctx, tiledb.Ctx):
        raise ValueError(f"`ctx` expected a TileDB Context object but saw {type(ctx)}")

    with tiledb.scope_ctx(ctx):
        _from_pandas(uri, dataframe, tiledb_args)


def _from_pandas(uri, dataframe, tiledb_args):
    import pandas as pd

    mode = tiledb_args.get("mode", "ingest")

    if mode != "append" and tiledb.array_exists(uri):
        raise tiledb.TileDBError(f"Array URI '{uri}' already exists!")

    sparse = tiledb_args["sparse"]
    index_dims = tiledb_args.get("index_dims") or ()
    row_start_idx = tiledb_args.get("row_start_idx")

    write = True
    create_array = True
    if mode is not None:
        if mode == "schema_only":
            write = False
        elif mode == "append":
            create_array = False
            schema = tiledb.ArraySchema.load(uri)
            if not schema.sparse and row_start_idx is None:
                raise tiledb.TileDBError(
                    "Cannot append to dense array without 'row_start_idx'"
                )
        elif mode != "ingest":
            raise tiledb.TileDBError(f"Invalid mode specified ('{mode}')")

    # TODO: disentangle the full_domain logic
    full_domain = tiledb_args.get("full_domain", False)
    if sparse is False and (not index_dims or "index_col" not in tiledb_args):
        full_domain = True
    if full_domain is None and tiledb_args.get("nrows"):
        full_domain = False

    date_spec = tiledb_args.get("date_spec")
    if date_spec:
        dataframe = dataframe.assign(
            **{
                name: pd.to_datetime(dataframe[name], format=format)
                for name, format in date_spec.items()
            }
        )

    dataframe.columns = dataframe.columns.map(str)
    column_infos = _get_column_infos(
        dataframe, tiledb_args.get("column_types"), tiledb_args.get("varlen_types")
    )

    with tiledb.scope_ctx(tiledb_args.get("ctx")):
        if create_array:
            with warnings.catch_warnings():
                warnings.simplefilter("always")
                _create_array(
                    uri,
                    dataframe,
                    sparse,
                    full_domain,
                    index_dims,
                    column_infos,
                    tiledb_args,
                )

        if write:
            if tiledb_args.get("debug", True):
                print(f"`tiledb.from_pandas` writing '{len(dataframe)}' rows")

            write_dict, nullmaps = _df_to_np_arrays(
                dataframe, column_infos, tiledb_args.get("fillna")
            )
            _write_array(
                uri,
                dataframe,
                write_dict,
                nullmaps,
                create_array,
                index_dims,
                row_start_idx,
                timestamp=tiledb_args.get("timestamp"),
            )


def _create_array(uri, df, sparse, full_domain, index_dims, column_infos, tiledb_args):
    dims, dim_metadata = create_dims(
        df,
        index_dims,
        full_domain=full_domain,
        tile=tiledb_args.get("tile"),
        filters=tiledb_args.get("dim_filters", True),
    )
    sparse = _sparse_from_dtypes(dim_metadata.values(), sparse)

    # ignore any column used as a dim/index
    attr_names = [c for c in df.columns if c not in index_dims]
    attrs, attr_metadata = _get_attrs(
        attr_names, column_infos, tiledb_args.get("attr_filters", True)
    )
    enums = _get_enums(attr_names, column_infos)

    # create the ArraySchema
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        coord_filter = tiledb_args.get("coords_filters", True)
        schema = tiledb.ArraySchema(
            sparse=sparse,
            domain=tiledb.Domain(*dims),
            attrs=attrs,
            enums=enums,
            cell_order=tiledb_args["cell_order"],
            tile_order=tiledb_args["tile_order"],
            coords_filters=None
            if coord_filter is True
            else _get_schema_filters(coord_filter),
            offsets_filters=_get_schema_filters(
                tiledb_args.get("offsets_filters", True)
            ),
            # 0 will use the libtiledb internal default
            capacity=tiledb_args.get("capacity") or 0,
            # don't set allows_duplicates=True for dense
            allows_duplicates=sparse and tiledb_args.get("allows_duplicates", False),
        )

    tiledb.Array.create(uri, schema)

    # write the metadata so we can reconstruct df
    with tiledb.open(uri, "w") as A:
        write_array_metadata(A, attr_metadata, dim_metadata)


def _write_array(
    uri,
    df,
    write_dict,
    nullmaps,
    create_array,
    index_dims,
    row_start_idx=None,
    timestamp=None,
):

    with tiledb.open(uri, "w", timestamp=timestamp) as A:
        for j in range(A.schema.nattr):
            attr = A.schema.attr(j)
            if attr.enum_label is not None:
                enmr = A.enum(attr.enum_label).values()
                df[attr.name] = df[attr.name].cat.set_categories(enmr)
                write_dict[attr.name] = df[attr.name].cat.codes

        if A.schema.sparse:
            coords = []
            for k in range(A.schema.ndim):
                dim_name = A.schema.domain.dim(k).name
                if (
                    (not create_array or dim_name in index_dims)
                    and dim_name not in df.index.names
                    and dim_name != "__tiledb_rows"
                ):
                    # this branch handles the situation where a user did not specify
                    # index_col and is using mode='append'. We would like to try writing
                    # with the columns corresponding to existing dimension name.
                    coords.append(write_dict.pop(dim_name))
                else:
                    coords.append(df.index.get_level_values(k))
            # TODO ensure correct col/dim ordering
            tiledb.libtiledb._setitem_impl_sparse(
                A, tuple(coords), write_dict, nullmaps
            )

        else:
            if row_start_idx is None:
                row_start_idx = 0
            row_end_idx = row_start_idx + len(df)
            A._setitem_impl(slice(row_start_idx, row_end_idx), write_dict, nullmaps)


def open_dataframe(uri, *, attrs=None, use_arrow=None, idx=slice(None), ctx=None):
    """Open TileDB array at given URI as a Pandas dataframe

    If the array was saved using tiledb.from_pandas, then columns
    will be interpreted as non-primitive pandas or numpy types when
    available.

    :param uri:
    :return: dataframe constructed from given TileDB array URI

    **Example:**

    >>> import tiledb
    >>> df = tiledb.open_dataframe("iris.tldb")
    >>> tiledb.object_type("iris.tldb")
    'array'
    """
    check_dataframe_deps()

    # TODO support `distributed=True` option?
    with tiledb.open(uri, ctx=ctx) as A:
        df = A.query(attrs=attrs, use_arrow=use_arrow, coords=True).df[idx]

    if attrs and list(df.columns) != list(attrs):
        df = df[attrs]

    return df


def _iterate_csvs_pandas(csv_list, pandas_args):
    """Iterate over a list of CSV files. Uses pandas.read_csv with pandas_args and returns
    a list of dataframe(s) for each iteration, up to the specified 'chunksize' argument in
    'pandas_args'
    """
    import pandas as pd

    assert "chunksize" in pandas_args
    chunksize = pandas_args["chunksize"]

    rows_read = 0
    result_list = list()

    file_iter = iter(csv_list)
    next_file = next(file_iter, None)
    while next_file is not None:
        df_iter = pd.read_csv(next_file, **pandas_args)
        df_iter.chunksize = chunksize - rows_read

        df = next(df_iter, None)
        while df is not None:
            result_list.append(df)
            rows_read += len(df)
            df_iter.chunksize = chunksize - rows_read

            if rows_read == chunksize:
                yield result_list
                # start over
                rows_read = 0
                df_iter.chunksize = chunksize
                result_list = list()

            df = next(df_iter, None)

        next_file = next(file_iter, None)
        if next_file is None and len(result_list) > 0:
            yield result_list


def from_csv(uri: str, csv_file: Union[str, List[str]], **kwargs):
    """
    Create TileDB array at given URI from a CSV file or list of files

    :param uri: URI for new TileDB array
    :param csv_file: input CSV file or list of CSV files.
                     Note: multi-file ingestion requires a `chunksize` argument. Files will be read in batches of at least `chunksize` rows before writing to the TileDB array.

    :Keyword Arguments:

        * Any `pandas.read_csv <https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html>`_ supported keyword argument
        * **ctx** - A TileDB context
        * **sparse** - (default True) Create sparse schema
        * **index_dims** (``List[str]``) -- List of column name(s) to use as dimension(s) in TileDB array schema. This is the recommended way to create dimensions. (note: the Pandas ``read_csv`` argument ``index_col`` will be passed through if provided, which results in indexes that will be converted to dimnesions by default; however ``index_dims`` is preferred).
        * **allows_duplicates** - Generated schema should allow duplicates
        * **mode** - Creation mode, one of 'ingest' (default), 'schema_only', 'append'
        * **attr_filters** - FilterList to apply to Attributes: FilterList or Dict[str -> FilterList] for any attribute(s). Unspecified attributes will use default.
        * **dim_filters** - FilterList to apply to Dimensions: FilterList or Dict[str -> FilterList] for any dimensions(s). Unspecified dimensions will use default.
        * **offsets_filters** - FilterList to apply to all offsets
        * **full_domain** - Dimensions should be created with full range of the dtype
        * **tile** - Dimension tiling: accepts either an int that applies the tiling to all dimensions or a dict("dim_name": int) to specifically assign tiling to a given dimension
        * **row_start_idx** - Start index to start new write (for row-indexed ingestions).
        * **fillna** - Value to use to fill holes
        * **column_types** - Dictionary of {``column_name``: dtype} to apply dtypes to columns
        * **varlen_types** - A set of {dtypes}; any column wihin the set is converted to a variable length attribute
        * **capacity** - Schema capacity.
        * **date_spec** - Dictionary of {``column_name``: format_spec} to apply to date/time columns which are not correctly inferred by pandas 'parse_dates'. Format must be specified using the Python format codes: https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior
        * **cell_order** - (default 'row-major) Schema cell order: 'row-major', 'col-major', or 'hilbert'
        * **tile_order** - (default 'row-major) Schema tile order: 'row-major' or 'col-major'
        * **timestamp** - Write TileDB array at specific timestamp.

    :return: None

    **Example:**

    >>> import tiledb
    >>> tiledb.from_csv("iris.tldb", "iris.csv")
    >>> tiledb.object_type("iris.tldb")
    'array'
    """
    check_dataframe_deps()
    import pandas

    if "tiledb_args" in kwargs:
        tiledb_args = kwargs.get("tiledb_args")
    else:
        tiledb_args = parse_tiledb_kwargs(kwargs)

    multi_file = False
    pandas_args = copy.deepcopy(kwargs)

    ##########################################################################
    # set up common arguments
    ##########################################################################
    ctx = tiledb_args.get("ctx")
    if ctx is not None and not isinstance(ctx, tiledb.Ctx):
        raise ValueError(f"`ctx` expected a TileDB Context object but saw {type(ctx)}")

    if isinstance(csv_file, str) and not os.path.isfile(csv_file):
        # for non-local files, use TileDB VFS i/o
        vfs = tiledb.VFS(ctx=ctx)
        csv_file = tiledb.FileIO(vfs, csv_file, mode="rb")
    elif isinstance(csv_file, (list, tuple)):
        # TODO may be useful to support a filter callback here
        multi_file = True

    mode = tiledb_args.get("mode", None)
    if mode is not None:
        # For schema_only mode we need to pass a max read count into
        #   pandas.read_csv
        # Note that 'nrows' is a pandas arg!
        if mode == "schema_only" and "nrows" not in kwargs:
            pandas_args["nrows"] = 500
        elif mode not in ["ingest", "append"]:
            raise tiledb.TileDBError("Invalid mode specified ('{}')".format(mode))

    with tiledb.scope_ctx(ctx):
        if mode != "append" and tiledb.array_exists(uri):
            raise tiledb.TileDBError("Array URI '{}' already exists!".format(uri))

    # this is a pandas pass-through argument, do not pop!
    chunksize = kwargs.get("chunksize", None)

    if multi_file and not (chunksize or mode == "schema_only"):
        raise tiledb.TileDBError(
            "Multiple input CSV files requires a 'chunksize' argument"
        )

    if multi_file:
        input_csv_list = csv_file
    else:
        input_csv = csv_file

    ##########################################################################
    # handle multi_file and chunked arguments
    ##########################################################################
    # we need to use full-domain for multi or chunked reads, because we
    # won't get a chance to see the full range during schema creation
    if multi_file or chunksize is not None:
        if "nrows" not in kwargs:
            tiledb_args["full_domain"] = True

    ##########################################################################
    # read path
    ##########################################################################
    if multi_file:
        if mode == "append":
            pass

        rows_written = 0

        # multi-file or chunked always writes to full domain
        # TODO: allow specifying dimension range for schema creation
        tiledb_args["full_domain"] = True

        for df_list in _iterate_csvs_pandas(input_csv_list, pandas_args):
            if df_list is None:
                break
            df = pandas.concat(df_list)
            if "index_col" not in tiledb_args and df.index.name is None:
                df.index.name = "__tiledb_rows"

            tiledb_args["row_start_idx"] = rows_written

            from_pandas(uri, df, tiledb_args=tiledb_args, pandas_args=pandas_args)

            tiledb_args["mode"] = "append"
            rows_written += len(df)

            if mode == "schema_only":
                break

    elif chunksize is not None:
        rows_written = 0
        # for chunked reads, we need to iterate over chunks
        df_iter = pandas.read_csv(input_csv, **pandas_args)
        df = next(df_iter, None)
        while df is not None:
            if "index_col" not in tiledb_args and df.index.name is None:
                df.index.name = "__tiledb_rows"

            # tell from_pandas what row to start the next write
            tiledb_args["row_start_idx"] = rows_written

            from_pandas(uri, df, tiledb_args=tiledb_args, pandas_args=pandas_args)

            tiledb_args["mode"] = "append"
            rows_written += len(df)

            df = next(df_iter, None)

    else:
        df = pandas.read_csv(csv_file, **kwargs)
        if "index_col" not in tiledb_args and df.index.name is None:
            df.index.name = "__tiledb_rows"

        kwargs.update(tiledb_args)
        from_pandas(uri, df, **kwargs)
