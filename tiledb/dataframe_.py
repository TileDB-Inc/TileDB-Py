import tiledb, numpy as np
import json
import sys

if sys.version_info >= (3,3):
    unicode_type = str
else:
    unicode_type = unicode

# TODO
# - handle missing values
# - handle extended datatypes
# - implement distributed CSV import
# - implement support for read CSV via TileDB VFS from any supported FS

TILEDB_KWARG_DEFAULTS = {
    'ctx': tiledb.default_ctx(),
    'cell_order': 'row-major',
    'tile_order': 'row-major',
    'sparse': False
}

class ColumnInfo:
    def __init__(self, dtype, repr=None):
        self.dtype = dtype
        self.repr = repr

def attr_dtype_from_column(col):
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
        return ColumnInfo(unicode_type)

    if col_dtype == 'bool':
        return ColumnInfo(np.uint8, repr=np.dtype('bool'))

    if col_dtype == np.dtype("O"):
        # Note: this does a full scan of the column... not sure what else to do here
        #       because Pandas allows mixed string column types (and actually has
        #       problems w/ allowing non-string types in object columns)
        inferred_dtype = pd.api.types.infer_dtype(col)

        if inferred_dtype == 'bytes':
            return ColumnInfo(bytes)
        elif inferred_dtype == 'string':
            return ColumnInfo(unicode_type)

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
def attrs_from_df(df, ctx=None):
    attr_reprs = dict()

    if ctx is None:
        ctx = tiledb.default_ctx()
    attrs = list()
    for name, col in df.items():
        attr_info = attr_dtype_from_column(col)
        attrs.append(tiledb.Attr(name=name, dtype=attr_info.dtype))

        if attr_info.repr is not None:
            attr_reprs[name] = attr_info.repr

    return attrs, attr_reprs


def from_csv(uri, csv_file, distributed=False, **kwargs):

    try:
        import pandas
    except ImportError as exc:
        raise ImportError("tiledb.from_csv requires pandas") from exc

    tiledb_args = kwargs.pop('tiledb_args', None)

    df = pandas.read_csv(csv_file, **kwargs)
    from_dataframe(uri, df, tiledb_args=tiledb_args, **kwargs)


def write_attr_metadata(array, metadata):
    """
    :param array: open, writable TileDB array
    :param metadata: dict
    :return:
    """

    md_dict = {n: str(t) for n,t in metadata.items()}

    array.meta['__pandas_attribute_repr'] = json.dumps(md_dict)


def from_dataframe(uri, dataframe, **kwargs):

    args = TILEDB_KWARG_DEFAULTS
    tiledb_args = kwargs.pop('tiledb_args', None)
    if tiledb_args is not None:
        args.update(tiledb_args)

    ctx = args['ctx']
    tile_order = args['tile_order']
    cell_order = args['cell_order']
    sparse = args['sparse']

    nrows = len(dataframe)
    tiling = np.min((nrows % 200, nrows))

    # create the domain and attributes
    dim = tiledb.Dim(
        name="rows",
        domain=(0, nrows-1),
        dtype=np.uint64,
        tile=tiling,
    )
    domain = tiledb.Domain(
       dim,
       ctx = ctx
    )
    attrs, attr_metadata = attrs_from_df(dataframe)

    # now create the ArraySchema
    schema = tiledb.ArraySchema(
        domain=domain,
        attrs=attrs,
        cell_order=cell_order,
        tile_order=tile_order,
        sparse=sparse
    )

    tiledb.Array.create(uri, schema, ctx=ctx)

    with tiledb.DenseArray(uri, 'w') as A:
        write_dict =  {k: v.values for k,v in dataframe.to_dict(orient='series').items()}
        A[0:nrows] = write_dict

        if attr_metadata is not None:
            write_attr_metadata(A, attr_metadata)


def open_dataframe(uri):
    """

    :param uri:
    :return: dataframe constructed from given TileDB array URI
    """

    import pandas as pd

    with tiledb.open(uri) as A:
        if not '__pandas_attribute_repr' in A.meta:
            raise ValueError("Missing key '__pandas_attribute_repr'")
        repr_meta = json.loads(A.meta['__pandas_attribute_repr'])

        data = A[:]

        for col_name, col_val in data.items():
            if col_name in repr_meta:
                new_col = pd.Series(col_val, dtype=repr_meta[col_name])
                data[col_name] = new_col

    return pd.DataFrame.from_dict(data)
