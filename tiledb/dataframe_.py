import tiledb, numpy as np
import json
import sys
import os
import io
from collections import OrderedDict

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
    'cell_order': 'row-major',
    'tile_order': 'row-major',
    'sparse': False
}

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
def attrs_from_df(df, index_dims=None, ctx=None):
    attr_reprs = dict()

    if ctx is None:
        ctx = tiledb.default_ctx()
    attrs = list()
    for name, col in df.items():
        # ignore any column used as a dim/index
        if index_dims and name in index_dims:
            continue
        attr_info = dtype_from_column(col)
        attrs.append(tiledb.Attr(name=name, dtype=attr_info.dtype))

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
    """Create TileDB array at given URI from pandas dataframe

    :param uri: URI for new TileDB array
    :param dataframe: pandas DataFrame
    :param kwargs: optional keyword arguments for Pandas and TileDB.
                TileDB context and configuration arguments
                may be passed in a dictionary as `tiledb_args={...}`
    :return:
    """
    args = TILEDB_KWARG_DEFAULTS

    #tiledb_args = kwargs.pop('tiledb_args', {})
    #index_dims = tiledb_args.pop('index_dims', None)

    #if isinstance(tiledb_args, dict):
    #    args.update(tiledb_args)
    if 'ctx' not in kwargs:
        args['ctx'] = tiledb.default_ctx()
    if 'sparse' in kwargs:
        args['sparse'] = kwargs.pop('sparse', None)

    index_dims = kwargs.pop('index_dims', None)

    ctx = args['ctx']
    tile_order = args['tile_order']
    cell_order = args['cell_order']
    sparse = args['sparse']

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
    attrs, attr_metadata = attrs_from_df(dataframe, index_dims=index_dims)

    # now create the ArraySchema
    schema = tiledb.ArraySchema(
        domain=domain,
        attrs=attrs,
        cell_order=cell_order,
        tile_order=tile_order,
        sparse=sparse
    )

    tiledb.Array.create(uri, schema, ctx=ctx)

    write_dict = {k: v.values for k,v in dataframe.to_dict(orient='series').items()}

    index_metadata = get_index_metadata(dataframe)

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


def from_csv(uri, csv_file, distributed=False, **kwargs):
    """Create TileDB array at given URI from a CSV file

    :param uri: URI for new TileDB array
    :param csv_file: input CSV file
    :param distributed:
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

    tiledb_args = kwargs.pop('tiledb_args', {})

    if isinstance(csv_file, str) and not os.path.isfile(csv_file):
        # for non-local files, use TileDB VFS i/o
        ctx = tiledb_args.get('ctx', tiledb.default_ctx())
        vfs = tiledb.VFS(ctx=ctx)
        csv_file = tiledb.FileIO(vfs, csv_file, mode='rb')

    df = pandas.read_csv(csv_file, **kwargs)
    from_dataframe(uri, df, tiledb_args=tiledb_args, **kwargs)
