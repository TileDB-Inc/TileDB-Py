import tiledb, numpy as np
import json
import sys
import os
import io
from collections import OrderedDict
import warnings

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
    'cell_order': 'row-major',
    'tile_order': 'row-major',
    'allows_duplicates': False,
    'sparse': False,
    'mode': 'ingest',
    'attrs_filters': None,
    'coords_filters': None
}

def parse_tiledb_kwargs(kwargs):
    args = dict(TILEDB_KWARG_DEFAULTS)

    if 'ctx' in kwargs:
        args['ctx'] = kwargs.pop('ctx')
    if 'sparse' in kwargs:
        args['sparse'] = kwargs.pop('sparse')
    if 'index_dims' in kwargs:
        args['index_dims'] = kwargs.pop('index_dims')
    if 'allows_duplicates' in kwargs:
        args['allows_duplicates'] = kwargs.pop('allows_duplicates')
    if 'mode' in kwargs:
        args['mode'] = kwargs.pop('mode')
    if 'attrs_filters' in kwargs:
        args['attrs_filters'] = kwargs.pop('attrs_filters')
    if 'coords_filters' in kwargs:
        args['coords_filters'] = kwargs.pop('coords_filters')

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
                "Column '{}' has mixed value dtype and cannot yet be stored as a TileDB attribute"
            )

    raise ValueError(
        "Unhandled column type: '{}'".format(
            col_dtype
        )
    )

# TODO make this a staticmethod on Attr?
def attrs_from_df(df, index_dims=None, filters=None, ctx=None):
    attr_reprs = dict()

    if ctx is None:
        ctx = tiledb.default_ctx()

    attrs = list()
    for name, col in df.items():
        # ignore any column used as a dim/index
        if index_dims and name in index_dims:
            continue
        attr_info = dtype_from_column(col)
        attrs.append(tiledb.Attr(name=name, dtype=attr_info.dtype, filters=filters))

        if attr_info.repr is not None:
            attr_reprs[name] = attr_info.repr

    return attrs, attr_reprs

def dim_for_column(ctx, df, col, col_name):
    if isinstance(col, np.ndarray):
        col_values = col
    else:
        col_values = col.values

    if len(col_values) < 1:
        raise ValueError("Empty column '{}' cannot be used for dimension!".format(col_name))

    dim_info = dtype_from_column(col_values)

    if col_values.dtype is np.dtype('O'):
        if type(col_values[0]) in (bytes, unicode_type):
            dim_min, dim_max = (None, None)
            # TODO... core only supports TILEDB_ASCII right now
            dim_info = ColumnInfo(np.bytes_)
        else:
            raise TypeError("other unknown column type not yet supported")
    else:
        dim_min = np.min(col_values)
        dim_max = np.max(col_values)

    dim = tiledb.Dim(
        name = col_name,
        domain = (dim_min, dim_max),
        dtype = dim_info.dtype,
        tile = 1 # TODO
    )

    return dim

def get_index_metadata(dataframe):
    md = dict()
    for index in dataframe.index.names:
        # Note: this may be expensive.
        md[index] = dtype_from_column(dataframe.index.get_level_values(index)).dtype

    return md

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

def create_dims(ctx, dataframe, index_dims):
    import pandas as pd
    index = dataframe.index
    index_dict = OrderedDict()

    if isinstance(index, pd.MultiIndex):
        for name in index.names:
            index_dict[name] = dataframe.index.get_level_values(name)

    elif isinstance(index, (pd.Index, pd.RangeIndex, pd.Int64Index)):
        if hasattr(index, 'name') and index.name is not None:
            name = index.name
        else:
            name = 'rows'

        index_dict[name] = index.values

    else:
        raise ValueError("Unhandled index type {}".format(type(index)))

    dims = list(
        dim_for_column(ctx, dataframe, values, name) for name,values in index_dict.items()
    )

    if index_dims:
        for name in index_dims:
            col = dataframe[name]
            dims.append(
                dim_for_column(ctx, dataframe, col.values, name)
            )

    return dims

def from_dataframe(uri, dataframe, **kwargs):
    # deprecated in 0.6.3
    warnings.warn("tiledb.from_dataframe is deprecated; please use .from_pandas",
                  DeprecationWarning)

    from_pandas(uri, dataframe, **kwargs)

def from_pandas(uri, dataframe, **kwargs):
    """Create TileDB array at given URI from pandas dataframe

    :param uri: URI for new TileDB array
    :param dataframe: pandas DataFrame
    :param mode: Creation mode, one of 'ingest' (default), 'create_schema'
    :param kwargs: optional keyword arguments for Pandas and TileDB.
        TileDB arguments: tile_order, cell_order, allows_duplicates, sparse,
                          mode, attrs_filters, coords_filters
    :return:
    """
    args = parse_tiledb_kwargs(kwargs)

    ctx = args.get('ctx', None)
    tile_order = args['tile_order']
    cell_order = args['cell_order']
    allows_duplicates = args['allows_duplicates']
    sparse = args['sparse']
    index_dims = args.get('index_dims', None)
    mode = args.get('mode', 'ingest')
    attrs_filters = args.get('attrs_filters', None)
    coords_filters = args.get('coords_filters', None)

    write = True
    if mode is not None and mode == 'schema_only':
        write = False

    if ctx is None:
        ctx = tiledb.default_ctx()

    if attrs_filters is None:
       attrs_filters = tiledb.FilterList(
            [tiledb.ZstdFilter(1, ctx=ctx)])

    if coords_filters is None:
        coords_filters = tiledb.FilterList(
            [tiledb.ZstdFilter(1, ctx=ctx)])

    nrows = len(dataframe)
    tiling = np.min((nrows % 200, nrows))

    # create the domain and attributes
    dims = create_dims(ctx, dataframe, index_dims)

    if len(dims) > 1:
        sparse = True
    if any([d.dtype in (np.bytes_, np.unicode_) for d in dims]):
        sparse = True
    if any([np.issubdtype(d.dtype, np.datetime64) for d in dims]):
        sparse = True

    domain = tiledb.Domain(
       *dims,
       ctx = ctx
    )

    attrs, attr_metadata = attrs_from_df(dataframe, index_dims=index_dims,
                                         filters=attrs_filters)

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

    write_dict = {k: v.values for k,v in dataframe.to_dict(orient='series').items()}

    index_metadata = get_index_metadata(dataframe)

    if write:
        try:
            A = tiledb.open(uri, 'w', ctx=ctx)

            if sparse:
                coords = []
                for k in range(len(dims)):
                    coords.append(dataframe.index.get_level_values(k))

                # TODO ensure correct col/dim ordering
                A[tuple(coords)] = write_dict

            else:
                A[:] = write_dict

            write_array_metadata(A, attr_metadata, index_metadata)
        finally:
            A.close()


def open_dataframe(uri):
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
    import pandas as pd

    # TODO support `distributed=True` option?

    with tiledb.open(uri) as A:
        #if not '__pandas_attribute_repr' in A.meta \
        #    and not '__pandas_repr' in A.meta:
        #    raise ValueError("Missing required keys to reload overloaded dataframe dtypes")

        # TODO missing key should only be a warning, return best-effort?
        # TODO this should be generalized for round-tripping overloadable types
        #      for any array (e.g. np.uint8 <> bool)
        repr_meta = None
        index_dims = None
        if '__pandas_attribute_repr' in A.meta:
            # backwards compatibility... unsure if necessary at this point
            repr_meta = json.loads(A.meta['__pandas_attribute_repr'])
        if '__pandas_index_dims' in A.meta:
            index_dims = json.loads(A.meta['__pandas_index_dims'])

        data = A[:]
        indexes = list()

        for col_name, col_val in data.items():
            if repr_meta and col_name in repr_meta:
                new_col = pd.Series(col_val, dtype=repr_meta[col_name])
                data[col_name] = new_col
            elif index_dims and col_name in index_dims:
                new_col = pd.Series(col_val, dtype=index_dims[col_name])
                data[col_name] = new_col
                indexes.append(col_name)

    new_df = pd.DataFrame.from_dict(data)
    if len(indexes) > 0:
        new_df.set_index(indexes, inplace=True)

    return new_df


def from_csv(uri, csv_file, **kwargs):
    """Create TileDB array at given URI from a CSV file

    :param uri: URI for new TileDB array
    :param csv_file: input CSV file
    :param kwargs: optional keyword arguments for Pandas and TileDB.
                TileDB context and configuration arguments
                may be passed in a dictionary as `tiledb_args={...}`
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

    mode = kwargs.pop('mode', None)
    tiledb_args = parse_tiledb_kwargs(kwargs)
    if mode is not None:
        tiledb_args['mode'] = mode
        # For schema-only mode we need to pass a max read count into
        #   pandas.read_csv
        # Note that 'nrows' is a pandas arg!
        if mode == 'schema_only' and not 'nrows' in kwargs:
            kwargs['nrows'] = 500

    if isinstance(csv_file, str) and not os.path.isfile(csv_file):
        # for non-local files, use TileDB VFS i/o
        ctx = tiledb_args.get('ctx', tiledb.default_ctx())
        vfs = tiledb.VFS(ctx=ctx)
        csv_file = tiledb.FileIO(vfs, csv_file, mode='rb')

    df = pandas.read_csv(csv_file, **kwargs)

    kwargs.update(tiledb_args)
    from_pandas(uri, df, **kwargs)