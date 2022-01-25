import numpy as np
import tiledb
import pytest
import hypothesis
import time
import tempfile
import os

import cc as lt

from common import paths_equal

# from tiledb.tests.fixtures
INTEGER_DTYPES = ["u1", "u2", "u4", "u8", "i1", "i2", "i4", "i8"]
STRING_DTYPES = ["U", "S"]
FLOAT_DTYPES = ["f4", "f8"]
SUPPORTED_DTYPES = INTEGER_DTYPES + STRING_DTYPES + FLOAT_DTYPES


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
    assert ctx.config == cfg

# NOMERGE
@pytest.fixture(scope="function", autouse=True)
def no_output(capfd):
    pass

def make_range(dtype):
    if np.issubdtype(dtype, np.number):
        return np.array([0,100.123]).astype(dtype), np.array([1]).astype(dtype)
    elif np.issubdtype(dtype, str) or np.issubdtype(dtype, bytes):
        return np.array(["a", "z"]).astype(dtype), None
    else:
        raise TypeError(f"Unsupported dtype '{dtype}'")

@pytest.mark.parametrize("dtype_str", SUPPORTED_DTYPES)
def test_dimension(dtype_str):
    if dtype_str == "U":
        # TODO this should assert TileDBError and continue
        pytest.skip("dtype('U') not supported for dimension")

    ctx = lt.Context()

    dtype = np.dtype(dtype_str)

    # TODO move this to pybind11
    tiledb_datatype = lt.DataType(tiledb.libtiledb.dtype_to_tiledb(dtype))

    range,extent = make_range(dtype)

    if dtype_str == "S":
        tiledb_datatype = lt.DataType.STRING_ASCII
        extent = np.array([], dtype=dtype) # null extent

    dim = lt.Dimension.create(ctx, "foo", tiledb_datatype, range, extent)
    print(dim)

def test_enums():
    def check_enum(name):
        """Helper function to iterate over a pybind11 enum and check that the typed value matches name"""
        enum_type = getattr(lt, name)
        for name,enum in enum_type.__members__.items():
            if name == "NONE":
                assert lt._enum_string(enum) == "NOOP"
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

    ctx = lt.Context()
    arr = lt.Array(ctx, uri, lt.QueryType.READ)
    assert arr.is_open()
    assert paths_equal(arr.uri(), uri)
    assert arr.schema == arr.schema

    # TODO test
    # open(tiledb_query_type_t query_type, tiledb_encryption_type_t encryption_type, const std::string& encryption_key, uint64_t timestamp)

    arr.reopen()
    arr.set_open_timestamp_start(0)
    arr.set_open_timestamp_end(1)
    arr.reopen()
    assert arr.open_timestamp_start == 0
    assert arr.open_timestamp_end == 1

    config = lt.Config({'foo': 'bar'})
    arr.set_config(config)
    assert arr.config()['foo'] == 'bar'

    arr.close()
    assert not arr.is_open()

    arr = lt.Array(ctx, uri, lt.QueryType.READ)

    # TODO test
    # consolidate sig1
    # consolidate sig2
    # vacuum
    # load_schema
    # create
    lt.Array.encryption_type(ctx, uri) == lt.EncryptionType.NO_ENCRYPTION
    # TODO assert lt.Array.load_schema(ctx, uri) == arr.schema
    assert arr.query_type() == lt.QueryType.READ

    arr.close()
    ####
    arrw = lt.Array(ctx, uri, lt.QueryType.WRITE)

    data = b'abcdef'
    arrw.put_metadata("key", lt.DataType.STRING_ASCII, data)
    arrw.close()

    arr = lt.Array(ctx, uri, lt.QueryType.READ)
    assert arr.metadata_num() == 1
    assert arr.has_metadata("key")
    mv = arr.get_metadata("key")
    assert bytes(mv) == data

    assert arr.get_metadata_from_index(0)[0] == lt.DataType.STRING_ASCII
    mv = arr.get_metadata_from_index(0)[1]
    assert bytes(mv) == data
    with pytest.raises(lt.TileDBError):
        arr.get_metadata_from_index(1)
    arr.close()

    arrw = lt.Array(ctx, uri, lt.QueryType.WRITE)
    arrw.delete_metadata("key")
    arrw.close()

    arr = lt.Array(ctx, uri, lt.QueryType.READ)
    with pytest.raises(KeyError):
        arr.get_metadata("key")
    assert not arr.has_metadata("key")[0]
    arr.close()

def test_domain():
    ctx = lt.Context()
    dom = lt.Domain(ctx)
    dim = lt.Dimension.create(ctx, "foo",
                              lt.DataType.INT32, np.int32([0,9]),
                              np.int32([9]))
    dom.add_dimension(dim)

    assert dom.datatype() == lt.DataType.INT32
    assert dom.cell_num() == 10
    # TODO assert dom.dimension("foo").domain() == ??? np.array?

def test_attribute():
    ctx = lt.Context()
    attr = lt.Attribute(ctx, "a1", lt.DataType.FLOAT64)

    assert(attr.name() == "a1")
    assert(attr.type() == lt.DataType.FLOAT64)
    assert(attr.cell_size() == 8)
    assert(attr.cell_val_num() == 1)
    attr.set_cell_val_num(5)
    assert(attr.cell_val_num() == 5)
    assert(attr.nullable() == False)
    attr.set_nullable(True)
    assert(attr.nullable() == True)
    #print(attr.filter_list()) # TODO


def test_schema():
    ctx = lt.Context()

    schema = lt.ArraySchema(ctx, lt.ArrayType.SPARSE)
    assert schema.array_type() == lt.ArrayType.SPARSE

    #TODO schema.dump()

    schema.set_capacity(101)
    assert schema.capacity() == 101

    schema.set_allows_dups(True)
    assert schema.allows_dups()

    with pytest.raises(lt.TileDBError):
        schema.set_tile_order(lt.LayoutType.HILBERT)
    schema.set_tile_order(lt.LayoutType.UNORDERED)
    assert schema.tile_order() == lt.LayoutType.UNORDERED

    # TODO schema.set_coords_filter_list(...)
    # TODO assert schema.coords_filter_list() == lt.FilterListType.NONE
    # TODO schema.set_offsets_filter_list
    # TODO assert schema.offsets_filter_list ==

    dom = lt.Domain(ctx)
    dim = lt.Dimension.create(ctx, "foo",
                              lt.DataType.INT32, np.int32([0,9]),
                              np.int32([9]))
    dom.add_dimension(dim)

    schema.set_domain(dom)
    # TODO dom and dimension need full equality check
    assert schema.domain().dimension("foo").name() == dim.name()

def test_query():
    def create_schema():
        schema = lt.ArraySchema(ctx, lt.ArrayType.SPARSE)
        dom = lt.Domain(ctx)
        dim = lt.Dimension.create(ctx, "foo",
                                  lt.DataType.STRING_ASCII, np.uint8([]),
                                  np.uint8([]))
        dom.add_dimension(dim)

        schema.set_domain(dom)
        return schema

    uri = tempfile.mkdtemp()

    ctx = lt.Context()
    schema = create_schema()
    lt.Array.create(uri, schema)
    arr = lt.Array(ctx, uri, lt.QueryType.READ)

    q = lt.Query(ctx, arr, lt.QueryType.READ)
    assert q.query_type() == lt.QueryType.READ

    q.add_range(0, "start", "end")