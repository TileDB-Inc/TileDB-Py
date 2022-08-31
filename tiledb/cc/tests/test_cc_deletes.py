#%%
from tkinter import W
import tiledb
import tiledb.cc as lt

import numpy as np

import tempfile
import time

global debug

#%%
def create_array(uri):
    schema = tiledb.ArraySchema(
        domain=tiledb.Domain(
            *[
                tiledb.Dim(
                    name="idx",
                    domain=(-10, 10),
                    tile=2,
                    dtype="float64",
                )
            ]
        ),
        attrs=[tiledb.Attr(name="data", dtype="float64", var=False)],
        cell_order="row-major",
        tile_order="row-major",
        capacity=10000,
        sparse=True,
        allows_duplicates=False,  # TODO parametrize
    )
    tiledb.SparseArray.create(uri, schema)


def write_array(uri, idx, data, timestamp=None):
    with tiledb.open(uri, "w", timestamp=timestamp) as A:
        A[idx] = {"data": data}


def apply_delete(uri, cond, timestamp=None):
    pyschema = tiledb.ArraySchema.load(uri)

    pyqc = tiledb.QueryCondition(cond)
    pyqc.init_query_condition(pyschema, ["data"])

    ctx = lt.Context()
    arr = lt.Array(ctx, uri, lt.QueryType.DELETE)
    if timestamp:
        arr = lt.Array(ctx, uri, lt.QueryType.DELETE, timestamp)
    else:
        arr = lt.Array(ctx, uri, lt.QueryType.DELETE)

    q = lt.Query(ctx, arr, lt.QueryType.DELETE)
    q.set_condition(pyqc)

    assert q.submit() == lt.QueryStatus.COMPLETE


# test_sparse_delete()
#%%
def test_sparse_delete_purge():
    def pa(uri, timestamp):
        print(f"<< read at: {timestamp}")
        with tiledb.open(uri, timestamp=timestamp) as A:
            print(A[:]["data"])

    uri = tempfile.mkdtemp()
    print(f"uri: {uri}")

    def p_all():
        pa(uri, 1)
        pa(uri, 2)
        pa(uri, 3)
        pa(uri, 4)
        pa(uri, None)

    def rr(ts):
        with tiledb.open(uri, timestamp=ts) as A:
            return A[:]["data"]

    ########################### WRITE
    create_array(uri)

    write_array(uri, idx=np.linspace(-10, -1, 5), data=np.arange(0, 5), timestamp=1)

    write_array(uri, idx=np.linspace(0, 10, 5), data=np.arange(5, 10), timestamp=2)
    assert 2 in rr(2)
    assert 2 in rr(3)

    ###########################

    print("---- initial array, timestamps 1,2")
    p_all()

    apply_delete(uri, "data == 2", timestamp=3)
    print("---- apply delete for 'data == 2' at ts==3")

    assert 2 in rr(2)
    assert 2 not in rr(3)

    ###########################

    time.sleep(0.01)
    tiledb.consolidate(uri)
    print("---- consolidate")

    assert 2 in rr(2)
    assert 2 not in rr(3)

    p_all()

    time.sleep(0.01)
    print("---- vacuum")
    tiledb.vacuum(uri)
    assert 2 in rr(2)
    assert 2 not in rr(3)

    p_all()

    time.sleep(0.01)
    cfg = tiledb.Config({"sm.consolidation.purge_deleted_cells": "true"})

    with tiledb.scope_ctx(cfg):
        tiledb.consolidate(uri, config=cfg)
        print("---- consolidate WITH PURGE")
    assert 2 in rr(2)
    assert 2 not in rr(3)

    p_all()

    tiledb.vacuum(uri)
    print("---- vacuum")
    assert 2 in rr(2)
    assert 2 not in rr(3)

    p_all()


test_sparse_delete_purge()


# %%
#%%
# def test_sparse_delete():
#    uri = tempfile.mkdtemp()
#    print(f"uri: {uri}")
#
#    write_array(uri)
#    with tiledb.open(uri) as A:
#        print(A[:]["data"])
#
#
#    pyschema = tiledb.ArraySchema.load(uri)
#
#    global pyqc
#    def apply_delete(uri, i):
#        pyqc = tiledb.QueryCondition(f"data == {i} or data == 5")
#        pyqc.init_query_condition(pyschema, ["data"])
#
#        ctx = lt.Context()
#        arr = lt.Array(ctx, uri, lt.QueryType.DELETE)
#
#        q = lt.Query(ctx, arr, lt.QueryType.DELETE)
#        q.set_condition(pyqc)
#
#        assert q.submit() == lt.QueryStatus.COMPLETE
#
#    apply_delete(uri, 2)
#    apply_delete(uri, 3)
#    apply_delete(uri, -2)
#    with tiledb.open(uri) as A:
#        print(A[:]["data"])
#
#    idx = np.arange(0, 10)
#    data = np.linspace(0, 2, 10)
#
#    with tiledb.open(uri, "w") as A:
#        A[idx] = {"data": idx[:5]}
#
#    with tiledb.open(uri) as A:
#        print(A[:]["data"])
#
#    apply_delete(uri, 7)
#    with tiledb.open(uri) as A:
#        print(A[:]["data"])
