import json
import sys
import os
import io
import copy
from collections import OrderedDict
import warnings

import numpy as np
import tiledb
from tiledb import TileDBError
#from tiledb.tests.common import xprint

if sys.version_info >= (3,3):
    unicode_type = str
else:
    unicode_type = unicode
unicode_dtype = np.dtype(unicode_type)

# TODO
# - handle missing values
# - handle extended datatypes

TILEDB_KWARG_DEFAULTS = {
    'ctx': None,
    'sparse': True,
    'index_dims': None,
    'allows_duplicates': True,
    'mode': 'ingest',
    'attrs_filters': None,
    'coords_filters': None,
    'full_domain': False,
    'tile': None,
    'row_start_idx': None,
    'fillna': None,
    'column_types': None,
    'capacity': None,
    'date_spec': None,
    'cell_order': 'row-major',
    'tile_order': 'row-major',
    'debug': None,
}

def parse_tiledb_kwargs(kwargs):
    parsed_args = dict(TILEDB_KWARG_DEFAULTS)

    for key in TILEDB_KWARG_DEFAULTS.keys():
        if key in kwargs:
            parsed_args[key] = kwargs.pop(key)

    return parsed_args

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
        index_md_name = index
        if index == None:
            index_md_name = '__tiledb_rows'
        # Note: this may be expensive.
        md[index_md_name] = dtype_from_column(dataframe.index.get_level_values(index)).dtype

    return md

def create_dims(ctx, dataframe, index_dims,
                tile=None, full_domain=False, sparse=None):
    import pandas as pd
    index = dataframe.index
    index_dict = OrderedDict()
    index_dtype = None

    per_dim_tile = False
    if tile is not None:
        if isinstance(tile, dict):
            per_dim_tile = True

        # input check, can't do until after per_dim_tile
        if (per_dim_tile and not all(map(lambda x: isinstance(x,(int,float)), tile.values()))) or \
           (per_dim_tile is False and not isinstance(tile, (int,float))):
            raise ValueError("Invalid tile kwarg: expected int or tuple of ints "
                             "got '{}'".format(tile))

    if isinstance(index, pd.MultiIndex):
        for name in index.names:
            index_dict[name] = dataframe.index.get_level_values(name)

    elif isinstance(index, (pd.Index, pd.RangeIndex, pd.Int64Index)):
        if hasattr(index, 'name') and index.name is not None:
            name = index.name
        else:
            index_dtype = np.dtype('uint64')
            name = '__tiledb_rows'

        index_dict[name] = index.values

    else:
        raise ValueError("Unhandled index type {}".format(type(index)))

    # create list of dim types
    # we need to know all the types in order to validate before creating Dims
    dim_types = list()
    for idx,(name, values) in enumerate(index_dict.items()):
        if per_dim_tile and name in tile:
            dim_tile = tile[name]
        elif per_dim_tile:
            # in this case we fall back to the default
            dim_tile = None
        else:
            # in this case we use a scalar (type-checked earlier)
            dim_tile = tile

        dim_types.append(dim_info_for_column(ctx, dataframe, values,
                         tile=dim_tile, full_domain=full_domain,
                         index_dtype=index_dtype))

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

    dims = list()
    for idx, (name, values) in enumerate(index_dict.items()):
        if per_dim_tile and name in tile:
            dim_tile = tile[name]
        elif per_dim_tile:
            # in this case we fall back to the default
            dim_tile = None
        else:
            # in this case we use a scalar (type-checked earlier)
            dim_tile = tile

        dims.append(dim_for_column(ctx, name, dim_types[idx], values,
                    tile=dim_tile, full_domain=full_domain, ndim=ndim))

    if index_dims:
        for name in index_dims:
            if per_dim_tile and name in tile:
                dim_tile = tile[name]
            elif per_dim_tile:
                # in this case we fall back to the default
                dim_tile = None
            else:
                # in this case we use a scalar  (type-checked earlier)
                dim_tile = tile

            col = dataframe[name]
            dims.append(
                dim_for_column(ctx, dataframe, col.values, name, tile=dim_tile)
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

def dataframe_to_np_arrays(dataframe, fillna=None):
    import pandas as pd
    if hasattr(pd, 'StringDtype'):
        # version > 1.0. StringDtype introduced in pandas 1.0
        ret = dict()
        for k,v in dataframe.to_dict(orient='series').items():
            if pd.api.types.is_extension_array_dtype(v):
                if fillna is None or not k in fillna:
                    raise ValueError("Missing 'fillna' value for column '{}' with pandas extension dtype".format(k))
                ret[k] = v.to_numpy(na_value=fillna[k])
            else:
                ret[k] = v.to_numpy()
    else:
        # version < 1.0
        ret = {k: v.values for k,v in dataframe.to_dict(orient='series').items()}

    return ret

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

    if 'tiledb_args' in kwargs:
        tiledb_args = kwargs.pop('tiledb_args')
    else:
        tiledb_args = parse_tiledb_kwargs(kwargs)

    ctx = tiledb_args.get('ctx', None)
    tile_order = tiledb_args['tile_order']
    cell_order = tiledb_args['cell_order']
    allows_duplicates = tiledb_args.get('allows_duplicates', False)
    sparse = tiledb_args['sparse']
    index_dims = tiledb_args.get('index_dims', None)
    mode = tiledb_args.get('mode', 'ingest')
    attrs_filters = tiledb_args.get('attrs_filters', None)
    coords_filters = tiledb_args.get('coords_filters', None)
    full_domain = tiledb_args.get('full_domain', False)
    capacity = tiledb_args.get('capacity', False)
    tile = tiledb_args.get('tile', None)
    nrows = tiledb_args.get('nrows', None)
    row_start_idx = tiledb_args.get('row_start_idx', None)
    fillna = tiledb_args.get('fillna', None)
    date_spec = tiledb_args.get('date_spec', None)
    column_types = tiledb_args.get('column_types', None)

    write = True
    create_array = True
    if mode is not None:
        if mode == 'schema_only':
            write = False
        elif mode == 'append':
            create_array = False
        elif mode != 'ingest':
            raise TileDBError("Invalid mode specified ('{}')".format(mode))

    if capacity is None:
        capacity = 0 # this will use the libtiledb internal default

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
            capacity=capacity,
            sparse=sparse
        )

        tiledb.Array.create(uri, schema, ctx=ctx)

        tiledb_args['mode'] = 'append'

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

    # write the metadata so we can reconstruct dataframe
    if create_array:
        index_metadata = get_index_metadata(dataframe)
        with tiledb.open(uri, 'w', ctx=ctx) as A:
            write_array_metadata(A, attr_metadata, index_metadata)

    if write:
        write_dict = dataframe_to_np_arrays(dataframe, fillna=fillna)

        if tiledb_args.get('debug', True):
            print("`tiledb.read_pandas` writing '{}' rows".format(len(dataframe)))

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
    rename_cols = dict()

    for col_name, col_val in result_dict.items():
        if repr_meta and col_name in repr_meta:
            new_col = pd.Series(col_val, dtype=repr_meta[col_name])
            result_dict[col_name] = new_col
        elif index_dims and col_name in index_dims:
            new_col = pd.Series(col_val, dtype=index_dims[col_name])
            result_dict[col_name] = new_col
            if col_name == '__tiledb_rows':
                rename_cols['__tiledb_rows'] = None
                indexes.append(None)
            else:
                indexes.append(col_name)

    for col_key,col_name in rename_cols.items():
        result_dict[col_name] = result_dict.pop(col_key)

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
    >>> tiledb.object_type("iris.tldb")
    'array'
    """
    if ctx is None:
        ctx = tiledb.default_ctx()
    # TODO support `distributed=True` option?
    with tiledb.open(uri, ctx=ctx) as A:
        nonempty = A.nonempty_domain()
        data = A.multi_index.__getitem__(tuple(slice(s1, s2) for s1,s2 in nonempty))
        new_df = _tiledb_result_as_dataframe(A, data)

    return new_df

def _iterate_csvs_pandas(csv_list, pandas_args):
    """Iterate over a list of CSV files. Uses pandas.read_csv with pandas_args and returns
    a list of dataframe(s) for each iteration, up to the specified 'chunksize' argument in
    'pandas_args'
    """
    import pandas as pd

    assert('chunksize' in pandas_args)
    chunksize = pandas_args['chunksize']

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
            * ``tile``: Dimension tiling: accepts either Int or a list of Tuple[Int] with per-dimension
              'tile' arguments to apply to the generated ArraySchema.
            * ``capacity``: Schema capacity
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

    if 'tiledb_args' in kwargs:
        tiledb_args = kwargs.get('tiledb_args')
    else:
        tiledb_args = parse_tiledb_kwargs(kwargs)

    multi_file = False
    debug = tiledb_args.get('debug', False)

    pandas_args = copy.deepcopy(kwargs)

    ##########################################################################
    # set up common arguments
    ##########################################################################
    if isinstance(csv_file, str) and not os.path.isfile(csv_file):
        # for non-local files, use TileDB VFS i/o
        ctx = tiledb_args.get('ctx', tiledb.default_ctx())
        vfs = tiledb.VFS(ctx=ctx)
        csv_file = tiledb.FileIO(vfs, csv_file, mode='rb')
    elif isinstance(csv_file, (list, tuple)):
        # TODO may be useful to support a filter callback here
        multi_file = True

    mode = tiledb_args.get('mode', None)
    if mode is not None:
        # For schema_only mode we need to pass a max read count into
        #   pandas.read_csv
        # Note that 'nrows' is a pandas arg!
        if mode == 'schema_only' and not 'nrows' in kwargs:
            pandas_args['nrows'] = 500
        elif mode not in ['ingest', 'append']:
            raise TileDBError("Invalid mode specified ('{}')".format(mode))

    # this is a pandas pass-through argument, do not pop!
    chunksize = kwargs.get('chunksize', None)

    if multi_file and not (chunksize or mode == 'schema_only'):
        raise TileDBError("Multiple input CSV files requires a 'chunksize' argument")

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
        if not 'nrows' in kwargs:
            tiledb_args['full_domain'] = True

    ##########################################################################
    # read path
    ##########################################################################
    if multi_file:
        array_created = False
        if mode == 'append':
            array_created = True

        rows_written = 0

        # multi-file or chunked always writes to full domain
        # TODO: allow specifying dimension range for schema creation
        tiledb_args['full_domain'] = True

        for df_list in _iterate_csvs_pandas(input_csv_list, pandas_args):
            if df_list is None:
                break
            df = pandas.concat(df_list)
            tiledb_args['row_start_idx'] = rows_written

            from_pandas(uri, df, tiledb_args=tiledb_args, pandas_args=pandas_args)

            rows_written += len(df)

            if mode == 'schema_only':
                break

    elif chunksize is not None:
        rows_written = 0
        # for chunked reads, we need to iterate over chunks
        df_iter = pandas.read_csv(input_csv, **pandas_args)
        df = next(df_iter, None)
        while df is not None:
            # tell from_pandas what row to start the next write
            tiledb_args['row_start_idx'] = rows_written

            from_pandas(uri, df, tiledb_args=tiledb_args, pandas_args=pandas_args)

            tiledb_args['mode'] = 'append'
            rows_written += len(df)

            df = next(df_iter, None)

    else:
        df = pandas.read_csv(csv_file, **kwargs)

        kwargs.update(tiledb_args)
        from_pandas(uri, df, **kwargs)
