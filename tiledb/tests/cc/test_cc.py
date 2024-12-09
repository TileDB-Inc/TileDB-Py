import os
import tempfile

import numpy as np
import pytest

import tiledb
import tiledb.libtiledb as lt
from tiledb.datatypes import DataType
from tiledb.main import PyFragmentInfo


def test_config():
    cfg = lt.Config()

    cfg.set("abc", "def")
    assert cfg.get("abc") == "def"

    cfg2 = lt.Config({"abc": "def"})
    assert cfg == cfg2
    assert cfg != lt.Config()

    cfg["xyz"] = "123"
    assert cfg["xyz"] == "123"
    del cfg["xyz"]
    with pytest.raises(KeyError):
        cfg["xyz"]
    with pytest.raises(lt.TileDBError):
        # TODO should this be KeyError?
        cfg.get("xyz")

    cfg2.unset("abc")
    with pytest.raises(lt.TileDBError):
        # TODO should this be KeyError?
        cfg2.get("abc")


def test_config_io(tmp_path):
    cfg = lt.Config({"abc": "def"})
    path = tmp_path / "test.cfg"
    cfg.save_to_file(str(path))
    cfg2 = lt.Config(str(path))
    assert cfg == cfg2


def test_context():
    ctx = lt.Context()
    cfg = lt.Config()
    cfg["abc"] = "123"
    ctx = lt.Context(cfg)
    assert ctx.config() == cfg


@pytest.mark.parametrize(
    "dtype_str", ["u1", "u2", "u4", "u8", "i1", "i2", "i4", "i8", "f4", "f8"]
)
def test_dimension(dtype_str):
    ctx = lt.Context()
    dt = DataType.from_numpy(dtype_str)
    domain = np.array([0, 100.123], dt.np_dtype)
    lt.Dimension(ctx, "foo", dt.tiledb_type, domain, dt.cast_tile_extent(1))
    # test STRING_ASCII
    lt.Dimension(ctx, "bar", lt.DataType.STRING_ASCII, None, None)


def test_enums():
    def check_enum(name):
        """Helper function to iterate over a pybind11 enum and check that the typed value matches name"""
        enum_type = getattr(lt, name)
        for name, enum in enum_type.__members__.items():
            if name == "NONE":
                assert lt._enum_string(enum) == "NOOP"
            elif name == "DICTIONARY":
                assert lt._enum_string(enum) == "DICTIONARY_ENCODING"
            else:
                assert name == lt._enum_string(enum)

    check_enum("DataType")
    check_enum("ArrayType")
    check_enum("FilterType")
    check_enum("QueryStatus")


def test_array():
    uri = tempfile.mkdtemp()

    # TODO BOOTSTRAP
    tiledb.from_numpy(uri, np.random.rand(4)).close()
    # lt.from_numpy(uri, np.random.rand(4)).close()

    ctx = lt.Context()
    arr = lt.Array(ctx, uri, lt.QueryType.READ)
    assert arr._is_open()
    assert os.path.basename(arr._uri()) == os.path.basename(uri)
    assert arr._schema == arr._schema

    arr._reopen()
    arr._set_open_timestamp_start(0)
    arr._set_open_timestamp_end(1)
    arr._reopen()
    assert arr._open_timestamp_start == 0
    assert arr._open_timestamp_end == 1

    arr._close()
    assert not arr._is_open()

    arr = lt.Array(ctx, uri, lt.QueryType.READ)

    # TODO test
    # consolidate sig1
    # consolidate sig2
    # vacuum
    # load_schema
    # create
    lt.Array._encryption_type(ctx, uri) == lt.EncryptionType.NO_ENCRYPTION
    # TODO assert lt.Array.load_schema(ctx, uri) == arr._schema
    assert arr._query_type() == lt.QueryType.READ

    arr._close()
    ####
    arr = lt.Array(ctx, uri, lt.QueryType.WRITE)
    arr._set_open_timestamp_start(1)
    arr._set_open_timestamp_end(1)
    arr._close()

    arr._open(lt.QueryType.WRITE)
    data = b"abcdef"
    arr._put_metadata("key", lt.DataType.STRING_ASCII, len(data), data)
    arr._close()

    arr._set_open_timestamp_start(1)
    arr._set_open_timestamp_end(1)
    arr._open(lt.QueryType.READ)
    assert arr._metadata_num() == 1
    assert arr._has_metadata("key")
    mv = arr._get_metadata("key")
    assert mv == (data, lt.DataType.STRING_ASCII)

    assert arr._get_metadata_from_index(0)[0] == lt.DataType.STRING_ASCII
    mv = arr._get_metadata_from_index(0)[1]
    assert bytes(mv) == data
    with pytest.raises(lt.TileDBError):
        arr._get_metadata_from_index(1)
    arr._close()

    arr._open(lt.QueryType.WRITE)
    arr._set_open_timestamp_start(2)
    arr._set_open_timestamp_end(2)
    arr._delete_metadata("key")
    arr._close()

    arr._set_open_timestamp_start(3)
    arr._set_open_timestamp_end(3)
    arr._open(lt.QueryType.READ)
    with pytest.raises(KeyError):
        arr._get_metadata("key")
    assert not arr._has_metadata("key")
    arr._close()


def test_consolidate_fragments():
    uri = tempfile.mkdtemp()
    ctx = lt.Context()
    config = lt.Config()

    tiledb.from_numpy(uri, np.random.rand(4)).close()

    with tiledb.open(uri, "w") as A:
        A[:] = np.random.rand(4)

    with tiledb.open(uri, "w") as A:
        A[:] = np.random.rand(4)

    schema = tiledb.ArraySchema.load(uri, ctx=ctx)
    fragment_info = PyFragmentInfo(uri, schema, False, ctx)

    assert fragment_info.get_num_fragments() == 3

    uris = fragment_info.get_uri()
    # get fragment name form alone, not the full path(s) (the part of each uri after the last /)
    # https://github.com/TileDB-Inc/TileDB-Py/pull/1946
    uris = [uri.split("/")[-1] for uri in uris]

    lt.Array._consolidate_fragments(uri, ctx, uris, config)

    fragment_info = PyFragmentInfo(uri, schema, False, ctx)
    # Fragmentinfo doesn't see the consolidated range
    assert fragment_info.get_num_fragments() == 1


def test_array_config():
    uri = tempfile.mkdtemp()

    # TODO BOOTSTRAP
    tiledb.from_numpy(uri, np.random.rand(4)).close()

    ctx = lt.Context()
    arr = lt.Array(ctx, uri, lt.QueryType.READ)
    arr._close()

    # TODO update this after SC-26938
    config = lt.Config({"foo": "bar"})
    arr._set_config(config)
    assert arr._config()["foo"] == "bar"


def test_domain():
    ctx = lt.Context()
    dom = lt.Domain(ctx)
    dim = lt.Dimension(ctx, "foo", lt.DataType.INT32, np.int32([0, 9]), np.int32([9]))
    dom._add_dim(dim)

    assert dom._tiledb_dtype == lt.DataType.INT32
    assert dom._ncell == 10
    # TODO assert dom.dimension("foo").domain() == ??? np.array?


def test_attribute():
    ctx = lt.Context()
    attr = lt.Attribute(ctx, "a1", lt.DataType.FLOAT64)

    assert attr._name == "a1"
    assert attr._tiledb_dtype == lt.DataType.FLOAT64
    assert attr._cell_size == 8
    assert attr._ncell == 1
    attr._ncell = 5
    assert attr._ncell == 5
    assert attr._nullable is False
    attr._nullable = True
    assert attr._nullable is True


def test_filter():
    ctx = lt.Context()
    fl = lt.FilterList(ctx)

    fl._add_filter(lt.Filter(ctx, lt.FilterType.ZSTD))
    assert fl._filter(0)._type == lt.FilterType.ZSTD
    assert fl._nfilters() == 1

    bzip_filter = lt.Filter(ctx, lt.FilterType.BZIP2)
    bzip_filter._set_option(ctx, lt.FilterOption.COMPRESSION_LEVEL, 2)
    assert bzip_filter._get_option(ctx, lt.FilterOption.COMPRESSION_LEVEL) == 2

    fl._add_filter(bzip_filter)
    assert fl._filter(1)._type == lt.FilterType.BZIP2
    assert fl._nfilters() == 2

    fl._chunksize = 100000
    assert fl._chunksize == 100000


def test_schema():
    ctx = lt.Context()

    schema = lt.ArraySchema(ctx, lt.ArrayType.SPARSE)
    assert schema._array_type == lt.ArrayType.SPARSE

    schema._capacity = 101
    assert schema._capacity == 101

    schema._allows_dups = True
    assert schema._allows_dups

    with pytest.raises(lt.TileDBError):
        schema._tile_order = lt.LayoutType.HILBERT
    if lt.version() >= (2, 24, 0):
        with pytest.raises(lt.TileDBError):
            schema._tile_order = lt.LayoutType.UNORDERED
    schema._tile_order = lt.LayoutType.ROW_MAJOR
    assert schema._tile_order == lt.LayoutType.ROW_MAJOR

    # TODO schema._set_coords_filter_list(...)
    # TODO assert schema._coords_filter_list() == lt.FilterListType.NONE
    # TODO schema._set_offsets_filter_list
    # TODO assert schema._offsets_filter_list ==

    dom = lt.Domain(ctx)
    dim = lt.Dimension(ctx, "foo", lt.DataType.INT32, np.int32([0, 9]), np.int32([9]))
    dom._add_dim(dim)

    schema._domain = dom
    # TODO dom and dimension need full equality check
    assert schema._domain._dim("foo")._name == dim._name


def test_query_string():
    def create_schema():
        schema = lt.ArraySchema(ctx, lt.ArrayType.SPARSE)
        dom = lt.Domain(ctx)
        dim = lt.Dimension(ctx, "foo", lt.DataType.STRING_ASCII, None, None)
        dom._add_dim(dim)

        schema._domain = dom
        return schema

    uri = tempfile.mkdtemp()

    ctx = lt.Context()
    schema = create_schema()
    lt.Array._create(ctx, uri, schema)
    arr = lt.Array(ctx, uri, lt.QueryType.READ)

    q = lt.Query(ctx, arr, lt.QueryType.READ)
    assert q.query_type == lt.QueryType.READ

    subarray = lt.Subarray(ctx, arr)
    subarray._add_dim_range(0, ("start", "end"))
    q.set_subarray(subarray)


def test_write_sparse():
    def create_schema():
        ctx = lt.Context()
        schema = lt.ArraySchema(ctx, lt.ArrayType.SPARSE)
        dom = lt.Domain(ctx)
        dim = lt.Dimension(
            ctx, "x", lt.DataType.INT32, np.int32([0, 9]), np.int32([10])
        )
        dom._add_dim(dim)

        attr = lt.Attribute(ctx, "a", lt.DataType.INT32)
        schema._add_attr(attr)

        schema._domain = dom
        return schema

    coords = np.arange(10).astype(np.int32)
    data = np.random.randint(0, 10, 10).astype(np.int32)

    def write():
        uri = tempfile.mkdtemp()

        ctx = lt.Context()
        schema = create_schema()
        lt.Array._create(ctx, uri, schema)
        arr = lt.Array(ctx, uri, lt.QueryType.WRITE)

        q = lt.Query(ctx, arr, lt.QueryType.WRITE)
        q.layout = lt.LayoutType.UNORDERED
        assert q.query_type == lt.QueryType.WRITE

        q.set_data_buffer("a", data, len(data))
        q.set_data_buffer("x", coords, len(coords))

        assert q._submit() == lt.QueryStatus.COMPLETE

        return uri

    def read(uri):
        ctx = lt.Context()
        arr = lt.Array(ctx, uri, lt.QueryType.READ)

        q = lt.Query(ctx, arr, lt.QueryType.READ)
        q.layout = lt.LayoutType.ROW_MAJOR
        assert q.query_type == lt.QueryType.READ

        rcoords = np.zeros(10).astype(np.int32)
        rdata = np.zeros(10).astype(np.int32)

        q.set_data_buffer("a", rdata, len(rdata))
        q.set_data_buffer("x", rcoords, len(rcoords))

        assert q._submit() == lt.QueryStatus.COMPLETE
        assert np.all(rcoords == coords)
        assert np.all(rdata == data)

    uri = write()
    read(uri)


def test_write_dense():
    def create_schema():
        ctx = lt.Context()
        schema = lt.ArraySchema(ctx, lt.ArrayType.DENSE)
        dom = lt.Domain(ctx)
        dim = lt.Dimension(
            ctx, "x", lt.DataType.UINT64, np.uint64([0, 9]), np.uint64([10])
        )
        dom._add_dim(dim)

        attr = lt.Attribute(ctx, "a", lt.DataType.FLOAT32)
        schema._add_attr(attr)

        schema._domain = dom
        return schema

    data = np.random.randint(0, 10, 10).astype(np.float32)

    def write():
        uri = tempfile.mkdtemp()

        ctx = lt.Context()
        schema = create_schema()
        lt.Array._create(ctx, uri, schema)
        arr = lt.Array(ctx, uri, lt.QueryType.WRITE)

        subarray = lt.Subarray(ctx, arr)
        subarray._add_dim_range(0, (0, 9))

        q = lt.Query(ctx, arr, lt.QueryType.WRITE)
        q.layout = lt.LayoutType.ROW_MAJOR
        assert q.query_type == lt.QueryType.WRITE

        q.set_subarray(subarray)

        q.set_data_buffer("a", data, len(data))
        # q.set_data_buffer("x", coords, len(coords))

        assert q._submit() == lt.QueryStatus.COMPLETE

        return uri

    def read(uri):
        ctx = lt.Context()
        arr = lt.Array(ctx, uri, lt.QueryType.READ)

        subarray = lt.Subarray(ctx, arr)
        subarray._add_dim_range(0, (0, 9))

        q = lt.Query(ctx, arr, lt.QueryType.READ)
        q.layout = lt.LayoutType.ROW_MAJOR
        assert q.query_type == lt.QueryType.READ

        q.set_subarray(subarray)

        rdata = np.zeros(10).astype(np.float32)

        q.set_data_buffer("a", rdata, len(rdata))

        assert q._submit() == lt.QueryStatus.COMPLETE
        assert np.all(rdata == data)

    uri = write()
    read(uri)
