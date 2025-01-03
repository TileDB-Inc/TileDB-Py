import os
from pathlib import Path

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
    assert str(Path("tiledb") / "schema_evolution.cc") in str(excinfo.value)

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


@pytest.mark.skipif(
    tiledb.libtiledb.version() < (2, 27),
    reason="Dropping a fixed-sized attribute and adding it back"
    "as a var-sized attribute is not supported in TileDB < 2.27",
)
@pytest.mark.parametrize("dtype_str", ["S", "U"])
def test_schema_evolution_drop_fixed_attribute_and_add_back_as_var_sized(
    tmp_path, dtype_str
):
    ctx = tiledb.default_ctx()
    uri = str(tmp_path)
    attrs = [
        tiledb.Attr(name="a", dtype=np.int32),
        tiledb.Attr(name="b", dtype=np.int32),
    ]
    dims = [tiledb.Dim(domain=(1, 10), dtype=np.int32)]
    domain = tiledb.Domain(*dims)
    schema = tiledb.ArraySchema(domain=domain, attrs=attrs, sparse=False)
    tiledb.Array.create(uri, schema)

    original_data = np.arange(1, 11)
    with tiledb.open(uri, "w") as A:
        A[:] = {"a": original_data, "b": original_data}

    se = tiledb.ArraySchemaEvolution(ctx)
    se.drop_attribute("a")
    se.array_evolve(uri)

    # check schema after dropping attribute
    with tiledb.open(uri) as A:
        assert not A.schema.has_attr("a")
        assert A.schema.attr("b").dtype == np.int32

    se = tiledb.ArraySchemaEvolution(ctx)
    newattr = tiledb.Attr("a", dtype=dtype_str, var=True)
    se.add_attribute(newattr)
    se.array_evolve(uri)

    # check schema and data after adding attribute back as a var-sized attribute
    with tiledb.open(uri) as A:
        assert A.schema.has_attr("a")
        assert A.schema.attr("a").dtype == dtype_str
        assert A.schema.attr("b").dtype == np.int32
        # check that each value equals to the fill value of "a" attribute
        assert_array_equal(A[:]["a"], np.array([newattr.fill] * 10, dtype=dtype_str))
        # check that nothing has changed for the "b" attribute
        assert_array_equal(A[:]["b"], original_data)

    # add new data to the array
    new_data = np.array(
        ["tiledb-string-n.{}".format(i) for i in range(1, 11)], dtype=dtype_str
    )
    with tiledb.open(uri, "w") as A:
        A[:] = {"a": new_data, "b": original_data}

    # check data for both attributes
    with tiledb.open(uri) as A:
        res = A[:]
        assert_array_equal(res["a"], new_data)
        assert_array_equal(res["b"], original_data)
