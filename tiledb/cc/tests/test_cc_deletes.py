#%%
from tkinter import W
import tiledb
import tiledb.cc as lt

import numpy as np

import tempfile
import time

#%%
def write_array(uri, timestamp=None):
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
    data = np.arange(5)
    idx = np.arange(-10, 0, 2)

    with tiledb.open(uri, "w", timestamp=timestamp) as A:
        A[idx] = {"data": data}


# test_sparse_delete()
#%%
def test_sparse_delete_purge():
    def apply_delete(uri, cond, timestamp=None):
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

    ########################### WRITE

    write_array(uri, timestamp=1)
    data = np.arange(5, 10)
    idx = np.arange(0, 10, 2)

    with tiledb.open(uri, "w", timestamp=2) as A:
        A[idx] = {"data": data}

    breakpoint()
    ###########################

    print("---- initial array, timestamps 1,2")
    with tiledb.open(uri) as A:
        print(A[:]["data"])

    pyschema = tiledb.ArraySchema.load(uri)

    apply_delete(uri, "data == 2", timestamp=3)
    print("---- apply delete for 'data == 2' at ts==3")

    p_all()

    ###########################

    time.sleep(0.01)
    tiledb.consolidate(uri)
    print("---- consolidate")

    p_all()

    time.sleep(0.01)
    print("---- vacuum")
    tiledb.vacuum(uri)

    p_all()

    time.sleep(0.01)
    cfg = tiledb.Config({"sm.consolidation.purge_deleted_cells": "true"})

    with tiledb.scope_ctx(cfg):
        tiledb.consolidate(uri, config=cfg)
        print("---- consolidate WITH PURGE")

    p_all()

    tiledb.vacuum(uri)
    print("---- vacuum")
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
