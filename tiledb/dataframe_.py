import tiledb, numpy as np
import json
import sys
import os
import io
from collections import OrderedDict
import warnings

from tiledb import TileDBError

if sys.version_info >= (3,3):
    unicode_type = str
else:
    unicode_type = unicode
unicode_dtype = np.dtype(unicode_type)

# TODO
# - handle missing values
# - handle extended datatypes
# - implement distributed CSV import
# - implement support for read CSV via TileDB VFS from any supported FS

TILEDB_KWARG_DEFAULTS = {
    'ctx': None,
    'sparse': True,
    'index_dims': None,
    'allows_duplicates': False,
    'mode': 'ingest',
    'attrs_filters': None,
    'coords_filters': None,
    'full_domain': False,
    'tile': None,
    'row_start_idx': None,
    'fillna': None,
    'column_types': None,
    'date_spec': None,
    'cell_order': 'row-major',
    'tile_order': 'row-major',
    'debug': None,
}

def parse_tiledb_kwargs(kwargs):
    args = dict(TILEDB_KWARG_DEFAULTS)

    for key in TILEDB_KWARG_DEFAULTS.keys():
        if key in kwargs:
            args[key] = kwargs.pop(key)

    return args

class ColumnInfo:
    def __init__(self, dtype, repr=None):
        self.dtype = dtype
        self.repr = repr

def dtype_from_column(col):
    import pandas as pd

    col_dtype = col.dtype
    # TODO add more basic types here
    if col_dtype in (np.int32, np.int64, np.uint32, np.uint64, np.float, np.double,
                     np.uint8):
        return ColumnInfo(col_dtype)

    # TODO this seems kind of brittle
    if col_dtype.base == np.dtype('M8[ns]'):
        if col_dtype == np.dtype('datetime64[ns]'):
            return ColumnInfo(col_dtype)
        elif hasattr(col_dtype, 'tz'):
            raise ValueError("datetime with tz not yet supported")
        else:
            raise ValueError("unsupported datetime subtype ({})".format(type(col_dtype)))

    # Pandas 1.0 has StringDtype extension type
    if col_dtype.name == 'string':
        return ColumnInfo(unicode_dtype)

    if col_dtype == 'bool':
        return ColumnInfo(np.uint8, repr=np.dtype('bool'))

    if col_dtype == np.dtype("O"):
        # Note: this does a full scan of the column... not sure what else to do here
        #       because Pandas allows mixed string column types (and actually has
        #       problems w/ allowing non-string types in object columns)
        inferred_dtype = pd.api.types.infer_dtype(col)

        if inferred_dtype == 'bytes':
            return ColumnInfo(np.bytes_)

        elif inferred_dtype == 'string':
            # TODO we need to make sure this is actually convertible
            return ColumnInfo(unicode_dtype)

        elif inferred_dtype == 'mixed':
            raise ValueError(
                "Column '{}' has mixed value dtype and cannot yet be stored as a TileDB attribute".format(col.name)
            )

    raise ValueError(
        "Unhandled column type: '{}'".format(
            col_dtype
        )
    )

# TODO make this a staticmethod on Attr?
def attrs_from_df(df,
                  index_dims=None, filters=None,
                  column_types=None, ctx=None):
    attr_reprs = dict()
    if ctx is None:
        ctx = tiledb.default_ctx()

    if column_types is None:
        column_types = dict()

    attrs = list()
    for name, col in df.items():
        # ignore any column used as a dim/index
        if index_dims and name in index_dims:
            continue

        if name in column_types:
            spec_type = column_types[name]
            # Handle ExtensionDtype
            if hasattr(spec_type, 'type'):
                spec_type = spec_type.type
            attr_info = ColumnInfo(spec_type)
        else:
            attr_info = dtype_from_column(col)
        attrs.append(tiledb.Attr(name=name, dtype=attr_info.dtype, filters=filters))

        if attr_info.repr is not None:
            attr_reprs[name] = attr_info.repr

    return attrs, attr_reprs

def dim_info_for_column(ctx, df, col, tile=None, full_domain=False, index_dtype=None):

    if isinstance(col, np.ndarray):
        col_values = col
    else:
        col_values = col.values

    if len(col_values) < 1:
        raise ValueError("Empty column '{}' cannot be used for dimension!".format(col_name))

    if index_dtype is not None:
        dim_info = ColumnInfo(index_dtype)
    elif col_values.dtype is np.dtype('O'):
        col_val0_type = type(col_values[0])
        if col_val0_type in (bytes, unicode_type):
            # TODO... core only supports TILEDB_ASCII right now
            dim_info = ColumnInfo(np.bytes_)
        else:
            raise TypeError("Unknown column type not yet supported ('{}')".format(col_val0_type))
    else:
        dim_info = dtype_from_column(col_values)

    return dim_info

def dim_for_column(ctx, name, dim_info, col, tile=None, full_domain=False, ndim=None):
    if isinstance(col, np.ndarray):
        col_values = col
    else:
        col_values = col.values

    if tile is None:
        if ndim is None:
            raise TileDBError("Unexpected Nonetype ndim")

        if ndim == 1:
            tile = 10000
        elif ndim == 2:
            tile = 1000
        elif ndim == 3:
            tile = 100
        else:
            tile = 10

    dtype = dim_info.dtype

    if full_domain:
        if not dim_info.dtype in (np.bytes_, np.unicode):
            # Use the full type domain, deferring to the constructor
            (dtype_min, dtype_max) = tiledb.libtiledb.dtype_range(dim_info.dtype)

            dim_max = dtype_max
            if dtype.kind == 'M':
                date_unit = np.datetime_data(dtype)[0]
                dim_min = np.datetime64(dtype_min + 1, date_unit)
                tile_max = np.iinfo(np.uint64).max - tile
                if np.abs(np.uint64(dtype_max) - np.uint64(dtype_min)) > tile_max:
                    dim_max = np.datetime64(dtype_max - tile, date_unit)
            elif dtype is np.int64:
                dim_min = dtype_min + 1
            else:
                dim_min = dtype_min

            if dtype.kind != 'M' and np.issubdtype(dtype, np.integer):
                tile_max = np.iinfo(np.uint64).max - tile
                if np.abs(np.uint64(dtype_max) - np.uint64(dtype_min)) > tile_max:
                    dim_max = dtype_max - tile
        else:
            dim_min, dim_max = (None, None)

    else:
        dim_min = np.min(col_values)
        dim_max = np.max(col_values)

    if not dim_info.dtype in (np.bytes_, np.unicode):
        if np.issubdtype(dtype, np.integer):
            dim_range = np.uint64(np.abs(np.uint64(dim_max) - np.uint64(dim_min)))
            if dim_range < tile:
                tile = dim_range
        elif np.issubdtype(dtype, np.float64):
            dim_range = dim_max - dim_min
            if dim_range < tile:
                tile = np.ceil(dim_range)

    dim = tiledb.Dim(
        name = name,
        domain = (dim_min, dim_max),
        dtype = dim_info.dtype,
        tile = tile
    )

    return dim

def get_index_metadata(dataframe):
    md = dict()
    for index in dataframe.index.names:
        # Note: this may be expensive.
        md[index] = dtype_from_column(dataframe.index.get_level_values(index)).dtype

    return md

def create_dims(ctx, dataframe, index_dims,
                tile=None, full_domain=False, sparse=None):
    import pandas as pd
    index = dataframe.index
    index_dict = OrderedDict()
    index_dtype = None

    if isinstance(index, pd.MultiIndex):
        for name in index.names:
            index_dict[name] = dataframe.index.get_level_values(name)

    elif isinstance(index, (pd.Index, pd.RangeIndex, pd.Int64Index)):
        if hasattr(index, 'name') and index.name is not None:
            name = index.name
        else:
            index_dtype = np.dtype('uint64')
            name = 'rows'

        index_dict[name] = index.values

    else:
        raise ValueError("Unhandled index type {}".format(type(index)))

    dim_types = list(
        dim_info_for_column(ctx, dataframe, values,
                            tile=tile, full_domain=full_domain,
                            index_dtype=index_dtype)
        for values in index_dict.values()
    )

    if any([d.dtype in (np.bytes_, np.unicode_) for d in dim_types]):
        if sparse is False:
            raise TileDBError("Cannot create dense array with string-typed dimensions")
        elif sparse is None:
            sparse = True

    d0 = dim_types[0]
    if not all(d0.dtype == d.dtype for d in dim_types[1:]):
        if sparse is False:
            raise TileDBError("Cannot create dense array with heterogeneous dimension data types")
        elif sparse is None:
            sparse = True

    ndim = len(dim_types)

    dims = list(
        dim_for_column(ctx, name, dim_types[i], values,
                       tile=tile, full_domain=full_domain, ndim=ndim)
        for i, (name, values) in enumerate(index_dict.items())
    )

    if index_dims:
        for name in index_dims:
            col = dataframe[name]
            dims.append(
                dim_for_column(ctx, dataframe, col.values, name)
            )

    return dims, sparse

def write_array_metadata(array, attr_metadata = None, index_metadata = None):
    """
    :param array: open, writable TileDB array
    :param metadata: dict
    :return:
    """
    if attr_metadata:
        attr_md_dict = {n: str(t) for n,t in attr_metadata.items()}
        array.meta['__pandas_attribute_repr'] = json.dumps(attr_md_dict)

    if index_metadata:
        index_md_dict = {n: str(t) for n,t in index_metadata.items()}
        array.meta['__pandas_index_dims'] = json.dumps(index_md_dict)

def from_dataframe(uri, dataframe, **kwargs):
    # deprecated in 0.6.3
    warnings.warn("tiledb.from_dataframe is deprecated; please use .from_pandas",
                  DeprecationWarning)

    from_pandas(uri, dataframe, **kwargs)

def from_pandas(uri, dataframe, **kwargs):
    """Create TileDB array at given URI from pandas dataframe

    :param uri: URI for new TileDB array
    :param dataframe: pandas DataFrame
    :param mode: Creation mode, one of 'ingest' (default), 'schema_only', 'append'

    :Keyword Arguments: optional keyword arguments for TileDB, see ``tiledb.from_csv``.

    :raises: :py:exc:`tiledb.TileDBError`
    :return: None

    """
    import pandas as pd

    args = parse_tiledb_kwargs(kwargs)

    ctx = args.get('ctx', None)
    tile_order = args['tile_order']
    cell_order = args['cell_order']
    allows_duplicates = args.get('allows_duplicates', False)
    sparse = args['sparse']
    index_dims = args.get('index_dims', None)
    mode = args.get('mode', 'ingest')
    attrs_filters = args.get('attrs_filters', None)
    coords_filters = args.get('coords_filters', None)
    full_domain = args.get('full_domain', False)
    tile = args.get('tile', None)
    nrows = args.get('nrows', None)
    row_start_idx = args.get('row_start_idx', None)
    fillna = args.pop('fillna', None)
    date_spec = args.pop('date_spec', None)
    column_types = args.pop('column_types', None)

    write = True
    create_array = True
    if mode is not None:
        if mode == 'schema_only':
            write = False
        elif mode == 'append':
            create_array = False
        elif mode != 'ingest':
            raise TileDBError("Invalid mode specified ('{}')".format(mode))

    if ctx is None:
        ctx = tiledb.default_ctx()

    if create_array:
        if attrs_filters is None:
           attrs_filters = tiledb.FilterList(
                [tiledb.ZstdFilter(1, ctx=ctx)])

        if coords_filters is None:
            coords_filters = tiledb.FilterList(
                [tiledb.ZstdFilter(1, ctx=ctx)])

        if nrows:
            if full_domain is None:
                full_domain = False

        # create the domain and attributes
        # if sparse==None then this function may return a default based on types
        dims, sparse = create_dims(ctx, dataframe, index_dims, sparse=sparse,
                           tile=tile, full_domain=full_domain)

        domain = tiledb.Domain(
           *dims,
           ctx = ctx
        )

        attrs, attr_metadata = attrs_from_df(dataframe,
                                             index_dims=index_dims,
                                             filters=attrs_filters,
                                             column_types=column_types)

        # now create the ArraySchema
        schema = tiledb.ArraySchema(
            domain=domain,
            attrs=attrs,
            cell_order=cell_order,
            tile_order=tile_order,
            coords_filters=coords_filters,
            allows_duplicates=allows_duplicates,
            sparse=sparse
        )

        tiledb.Array.create(uri, schema, ctx=ctx)

    # apply fill replacements for NA values if specified
    if fillna is not None:
        dataframe.fillna(fillna, inplace=True)

    # apply custom datetime parsing to given {'column_name': format_spec} pairs
    # format_spec should be provied using Python format codes:
    #     https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior
    if date_spec is not None:
        if type(date_spec) is not dict:
            raise TypeError("Expected 'date_spec' to be a dict, got {}".format(type(date_spec)))
        for name, spec in date_spec.items():
            dataframe[name] = pd.to_datetime(dataframe[name], format=spec)

    if write:
        write_dict = {k: v.values for k,v in dataframe.to_dict(orient='series').items()}

        index_metadata = get_index_metadata(dataframe)

        try:
            A = tiledb.open(uri, 'w', ctx=ctx)

            if A.schema.sparse:
                coords = []
                for k in range(A.schema.ndim):
                    coords.append(dataframe.index.get_level_values(k))

                # TODO ensure correct col/dim ordering
                A[tuple(coords)] = write_dict

            else:
                if row_start_idx is None:
                    row_start_idx = 0
                row_end_idx = row_start_idx + len(dataframe)
                A[row_start_idx:row_end_idx] = write_dict

            if create_array:
                write_array_metadata(A, attr_metadata, index_metadata)
        finally:
            A.close()

def _tiledb_result_as_dataframe(readable_array, result_dict):
    import pandas as pd
    # TODO missing key in the rep map should only be a warning, return best-effort?
    # TODO this should be generalized for round-tripping overloadable types
    #      for any array (e.g. np.uint8 <> bool)
    repr_meta = None
    index_dims = None
    if '__pandas_attribute_repr' in readable_array.meta:
        # backwards compatibility
        repr_meta = json.loads(readable_array.meta['__pandas_attribute_repr'])
    if '__pandas_index_dims' in readable_array.meta:
        index_dims = json.loads(readable_array.meta['__pandas_index_dims'])

    indexes = list()

    for col_name, col_val in result_dict.items():
        if repr_meta and col_name in repr_meta:
            new_col = pd.Series(col_val, dtype=repr_meta[col_name])
            result_dict[col_name] = new_col
        elif index_dims and col_name in index_dims:
            new_col = pd.Series(col_val, dtype=index_dims[col_name])
            result_dict[col_name] = new_col
            indexes.append(col_name)

    df = pd.DataFrame.from_dict(result_dict)
    if len(indexes) > 0:
        df.set_index(indexes, inplace=True)

    return df

def open_dataframe(uri, ctx=None):
    """Open TileDB array at given URI as a Pandas dataframe

    If the array was saved using tiledb.from_dataframe, then columns
    will be interpreted as non-primitive pandas or numpy types when
    available.

    :param uri:
    :return: dataframe constructed from given TileDB array URI

    **Example:**

    >>> import tiledb
    >>> df = tiledb.open_dataframe("iris.tldb")
    >>> tiledb.objec_type("iris.tldb")
    'array'
    """
    warnings.warn("open_dataframe is deprecated and will be removed in the next release",
                  DeprecationWarning)

    if ctx is None:
        ctx = tiledb.default_ctx()

    # TODO support `distributed=True` option?

    with tiledb.open(uri, ctx=ctx) as A:
        data = A[:]
        new_df = _tiledb_result_as_dataframe(A, data)

    return new_df

def from_csv(uri, csv_file, **kwargs):
    """
    Create TileDB array at given URI from a CSV file or list of files

    :param uri: URI for new TileDB array
    :param csv_file: input CSV file or list of CSV files.
                     Note: multi-file ingestion requires a `chunksize` argument. Files will
                     be read in batches of at least `chunksize` rows before writing to the
                     TileDB array.

    :Keyword Arguments:
        - Any ``pandas.read_csv`` supported keyword argument.
        - TileDB-specific arguments:
            * ``allows_duplicates``: Generated schema should allow duplicates
            * ``cell_order``: Schema cell order
            * ``tile_order``: Schema tile order
            * ``mode``: (default ``ingest``), Ingestion mode: ``ingest``, ``schema_only``,
              ``append``
            * ``full_domain``: Dimensions should be created with full range of the dtype
            * ``attrs_filters``: FilterList to apply to all Attributes
            * ``coords_filters``: FilterList to apply to all coordinates (Dimensions)
            * ``sparse``: (default True) Create sparse schema
            * ``tile``: Schema tiling (capacity)
            * ``date_spec``: Dictionary of {``column_name``: format_spec} to apply to date/time
              columns which are not correctly inferred by pandas 'parse_dates'.
              Format must be specified using the Python format codes:
              https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior
    :return: None

    **Example:**

    >>> import tiledb
    >>> tiledb.from_csv("iris.tldb", "iris.csv")
    >>> tiledb.object_type("iris.tldb")
    'array'
    """
    try:
        import pandas
    except ImportError as exc:
        print("tiledb.from_csv requires pandas")
        raise

    tiledb_args = parse_tiledb_kwargs(kwargs)
    multi_file = False
    debug = tiledb_args.get('debug', False)

    if isinstance(csv_file, str) and not os.path.isfile(csv_file):
        # for non-local files, use TileDB VFS i/o
        ctx = tiledb_args.get('ctx', tiledb.default_ctx())
        vfs = tiledb.VFS(ctx=ctx)
        csv_file = tiledb.FileIO(vfs, csv_file, mode='rb')
    elif isinstance(csv_file, (list, tuple)):
        # TODO may be useful to support a callback here
        multi_file = True

    mode = kwargs.pop('mode', None)
    if mode is not None:
        tiledb_args['mode'] = mode
        # For schema_only mode we need to pass a max read count into
        #   pandas.read_csv
        # Note that 'nrows' is a pandas arg!
        if mode == 'schema_only' and not 'nrows' in kwargs:
            kwargs['nrows'] = 500
        elif mode not in ['ingest', 'append']:
            raise TileDBError("Invalid mode specified ('{}')".format(mode))

    chunksize = kwargs.get('chunksize', None)

    if multi_file and not chunksize:
        raise TileDBError("Multiple input CSV files requires a 'chunksize' argument")

    if chunksize is not None or multi_file:
        if not 'nrows' in kwargs:
            full_domain = True

        array_created = False
        if mode == 'schema_only':
            raise TileDBError("schema_only ingestion not supported for chunked read")
        elif mode == 'append':
            array_created = True

        csv_kwargs = kwargs.copy()
        kwargs.update(tiledb_args)

        if multi_file:
            input_csv_list = csv_file
            csv_kwargs.pop("chunksize")
        else:
            input_csv = csv_file

        keep_reading = True
        rows_written = 0
        csv_idx = 0
        df_iter = None

        while keep_reading:
            # if we have multiple files, read them until we hit row threshold
            if multi_file:
                rows_read = 0
                input_dfs = list()
                while rows_read < chunksize and keep_reading:
                    input_csv = input_csv_list[csv_idx]
                    df = pandas.read_csv(input_csv, **csv_kwargs)
                    input_dfs.append(df)

                    rows_read += len(df)
                    csv_idx += 1
                    keep_reading = csv_idx < len(input_csv_list)

                df = pandas.concat(input_dfs)

            else:
                if not df_iter:
                    df_iter = pandas.read_csv(input_csv, **csv_kwargs)
                df = next(df_iter, None)

            if df is None:
                break

            kwargs['row_start_idx'] = rows_written
            kwargs['full_domain'] = True
            if array_created:
                kwargs['mode'] = 'append'
            # after the first chunk, switch to append mode
            array_created = True

            if debug:
                print("`tiledb.read_csv` flushing '{}' rows ('{}' files)".format(
                    len(df), csv_idx))

            # now flush
            from_pandas(uri, df, **kwargs)
            rows_written += len(df)

            if mode == 'schema_only':
                break

    else:
        df = pandas.read_csv(csv_file, **kwargs)

        kwargs.update(tiledb_args)
        from_pandas(uri, df, **kwargs)
