import os

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import tiledb


def test_schema_evolution(tmp_path):
    ctx = tiledb.default_ctx()
    se = tiledb.ArraySchemaEvolution(ctx)

    uri = str(tmp_path)

    attrs = [
        tiledb.Attr(name="a1", dtype=np.float64),
        tiledb.Attr(name="a2", dtype=np.int32),
    ]
    dims = [tiledb.Dim(domain=(0, 3), dtype=np.uint64)]
    domain = tiledb.Domain(*dims)
    schema = tiledb.ArraySchema(domain=domain, attrs=attrs, sparse=False)
    tiledb.Array.create(uri, schema)

    data1 = {
        "a1": np.arange(5, 9),
        "a2": np.random.randint(0, 1e7, size=4).astype(np.int32),
    }

    with tiledb.open(uri, "w") as A:
        A[:] = data1

    with tiledb.open(uri) as A:
        res = A[:]
        assert_array_equal(res["a1"], data1["a1"])
        assert_array_equal(res["a2"], data1["a2"])
        assert "a3" not in res.keys()

    newattr = tiledb.Attr("a3", dtype=np.int8)
    se.add_attribute(newattr)

    with pytest.raises(tiledb.TileDBError) as excinfo:
        se.add_attribute(newattr)
    assert "Input attribute name is already there" in str(excinfo.value)
    assert "tiledb/schema_evolution.cc" in str(excinfo.value)

    se.array_evolve(uri)

    data2 = {
        "a1": np.arange(5, 9),
        "a2": np.random.randint(0, 1e7, size=4).astype(np.int32),
        "a3": np.random.randint(0, 255, size=4).astype(np.int8),
    }

    with tiledb.open(uri, "w") as A:
        A[:] = data2

    def test_it():
        with tiledb.open(uri) as A:
            res = A[:]
            assert_array_equal(res["a1"], data2["a1"])
            assert_array_equal(res["a2"], data2["a2"])
            assert_array_equal(res["a3"], data2["a3"])

    test_it()
    tiledb.consolidate(uri)
    test_it()

    se = tiledb.ArraySchemaEvolution(ctx)
    se.drop_attribute("a1")
    se.array_evolve(uri)

    data3 = {
        "a2": np.random.randint(0, 1e7, size=4).astype(np.int32),
        "a3": np.random.randint(0, 255, size=4).astype(np.int8),
    }

    def test_it2():
        with tiledb.open(uri) as A:
            res = A[:]
            assert "a1" not in res.keys()
            assert_array_equal(res["a2"], data3["a2"])
            assert_array_equal(res["a3"], data3["a3"])

    with tiledb.open(uri, "w") as A:
        A[:] = data3

    test_it2()
    tiledb.consolidate(uri)
    test_it2()


def test_schema_evolution_timestamp(tmp_path):
    ctx = tiledb.default_ctx()
    se = tiledb.ArraySchemaEvolution(ctx)
    vfs = tiledb.VFS()

    uri = str(tmp_path)
    schema_uri = os.path.join(uri, "__schema")

    attrs = [tiledb.Attr(name="a1", dtype=np.float64)]
    domain = tiledb.Domain(tiledb.Dim(domain=(0, 3), dtype=np.uint64))
    schema = tiledb.ArraySchema(domain=domain, attrs=attrs, sparse=False)
    tiledb.Array.create(uri, schema)

    def get_schema_timestamps(schema_uri):
        schema_files = filter(lambda x: "__enumerations" not in x, vfs.ls(schema_uri))
        return [int(os.path.basename(file).split("_")[2]) for file in schema_files]

    assert 123456789 not in get_schema_timestamps(schema_uri)

    newattr = tiledb.Attr("a2", dtype=np.int8)
    se.timestamp(123456789)
    se.add_attribute(newattr)
    se.array_evolve(uri)

    assert 123456789 in get_schema_timestamps(schema_uri)


def test_schema_evolution_with_enmr(tmp_path):
    ctx = tiledb.default_ctx()
    se = tiledb.ArraySchemaEvolution(ctx)

    uri = str(tmp_path)

    attrs = [
        tiledb.Attr(name="a1", dtype=np.float64),
        tiledb.Attr(name="a2", dtype=np.int32),
    ]
    dims = [tiledb.Dim(domain=(0, 3), dtype=np.uint64)]
    domain = tiledb.Domain(*dims)
    schema = tiledb.ArraySchema(domain=domain, attrs=attrs, sparse=False)
    tiledb.Array.create(uri, schema)

    data1 = {
        "a1": np.arange(5, 9),
        "a2": np.random.randint(0, 1e7, size=4).astype(np.int32),
    }

    with tiledb.open(uri, "w") as A:
        A[:] = data1

    with tiledb.open(uri) as A:
        assert not A.schema.has_attr("a3")

    newattr = tiledb.Attr("a3", dtype=np.int8, enum_label="e3")
    se.add_attribute(newattr)

    with pytest.raises(tiledb.TileDBError) as excinfo:
        se.array_evolve(uri)
    assert " Attribute refers to an unknown enumeration" in str(excinfo.value)

    se.add_enumeration(tiledb.Enumeration("e3", True, np.arange(0, 8)))
    se.array_evolve(uri)

    se = tiledb.ArraySchemaEvolution(ctx)

    with tiledb.open(uri) as A:
        assert A.schema.has_attr("a3")
        assert A.attr("a3").enum_label == "e3"

    se.drop_enumeration("e3")

    with pytest.raises(tiledb.TileDBError) as excinfo:
        se.array_evolve(uri)
    assert "Unable to drop enumeration" in str(excinfo.value)

    se.drop_attribute("a3")
    se.array_evolve(uri)

    with tiledb.open(uri) as A:
        assert not A.schema.has_attr("a3")


@pytest.mark.parametrize(
    "type,data",
    (
        ("int", [0]),
        ("bool", [True, False]),
        ("str", ["abc", "defghi", "jk"]),
        ("bytes", [b"abc", b"defghi", b"jk"]),
    ),
)
def test_schema_evolution_extend_enmr(tmp_path, type, data):
    uri = str(tmp_path)
    enmr = tiledb.Enumeration("e", True, dtype=type)
    attrs = [tiledb.Attr(name="a", dtype=int, enum_label="e")]
    domain = tiledb.Domain(tiledb.Dim(domain=(0, 3), dtype=np.uint64))
    schema = tiledb.ArraySchema(domain=domain, attrs=attrs, enums=[enmr])
    tiledb.Array.create(uri, schema)

    with tiledb.open(uri) as A:
        assert A.schema.has_attr("a")
        assert A.attr("a").enum_label == "e"
        assert A.enum("e") == enmr

    se = tiledb.ArraySchemaEvolution()
    updated_enmr = enmr.extend(data)
    se.extend_enumeration(updated_enmr)
    se.array_evolve(uri)

    with tiledb.open(uri) as A:
        assert A.schema.has_attr("a")
        assert A.attr("a").enum_label == "e"
        assert A.enum("e") == updated_enmr


def test_schema_evolution_extend_check_bad_type():
    enmr = tiledb.Enumeration("e", True, dtype=str)
    with pytest.raises(tiledb.TileDBError):
        enmr.extend([1, 2, 3])
    with pytest.raises(tiledb.TileDBError):
        enmr.extend([True, False])
    enmr.extend(["a", "b"])

    enmr = tiledb.Enumeration("e", True, dtype=int)
    with pytest.raises(tiledb.TileDBError):
        enmr.extend(["a", "b"])
    with pytest.raises(tiledb.TileDBError):
        enmr.extend([True, False])
    enmr.extend([1, 2, 3])

    enmr = tiledb.Enumeration("e", True, dtype=bool)
    with pytest.raises(tiledb.TileDBError):
        enmr.extend(["a", "b"])
    with pytest.raises(tiledb.TileDBError):
        enmr.extend([1, 2, 3])
    enmr.extend([True, False])
