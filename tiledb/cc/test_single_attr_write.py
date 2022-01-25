import numpy as np
#import tiledb
import hypothesis
import time
import tempfile
import os

import cc as lt

from common import paths_equal

import pytest

def test_write_sparse():
    ncells = 10_000
    coords = np.arange(ncells).astype(np.int32)
    data = np.random.randint(0,int(10e7),ncells).astype(np.int32)
    b_data = np.arange(ncells).astype(np.int32)

    def create_schema():
        ctx = lt.Context()
        schema = lt.ArraySchema(ctx, lt.ArrayType.SPARSE)
        dom = lt.Domain(ctx)
        dim = lt.Dimension.create(ctx, "x",
                                  lt.DataType.INT32,
                                  np.int32([0,ncells-1]),
                                  np.int32([ncells]))
        dom.add_dimension(dim)

        a1 = lt.Attribute(ctx, "a", lt.DataType.INT32)
        schema.add_attribute(a1)
        a2 = lt.Attribute(ctx, "b", lt.DataType.INT32)
        schema.add_attribute(a2)


        schema.set_domain(dom)
        return schema

    #def do_write(array, buf_name, buf_data):


    def write():
        uri = tempfile.mkdtemp()

        ctx = lt.Context()
        schema = create_schema()
        lt.Array.create(uri, schema)
        arr = lt.Array(ctx, uri, lt.QueryType.WRITE)

        q = lt.Query(ctx, arr, lt.QueryType.WRITE)
        q.set_layout(lt.LayoutType.UNORDERED)

        q.set_data_buffer("a", data)
        q.set_data_buffer("a", data)
        q.set_data_buffer("x", coords)
        q.set_continuation()

        assert(q.submit() == lt.QueryStatus.COMPLETE)
        #q.finalize()

        fragment_uri = q.fragment_uri(0)
        print("written fragment uri is: ", fragment_uri)

        # write 2
        q.set_fragment_uri(fragment_uri)
        q.unset_buffer("a")
        q.unset_buffer("x")

        q.set_data_buffer("b", b_data)
        assert(q.submit() == lt.QueryStatus.COMPLETE)
        q.finalize()

        #q2 = lt.Query(ctx, arr, lt.QueryType.WRITE)
        #q2.set_layout(lt.LayoutType.UNORDERED)
        ##q.set_fragment_uri(fragment_uri)

        #q2.set_data_buffer("b", b_data)
        ##q2.set_data_buffer("x", coords)
        #assert(q2.submit() == lt.QueryStatus.COMPLETE)
        #q2.finalize()

        return uri

    def read(uri):
        ctx = lt.Context()
        arr = lt.Array(ctx, uri, lt.QueryType.READ)

        q = lt.Query(ctx, arr, lt.QueryType.READ)
        q.set_layout(lt.LayoutType.ROW_MAJOR)
        assert q.query_type() == lt.QueryType.READ

        rcoords = np.zeros(ncells).astype(np.int32)
        rdata_a = np.zeros(ncells).astype(np.int32)
        rdata_b = np.zeros(ncells).astype(np.int32)

        q.set_data_buffer("x", rcoords)
        q.set_data_buffer("a", rdata_a)
        q.set_data_buffer("b", rdata_b)

        breakpoint()

        assert(q.submit() == lt.QueryStatus.COMPLETE)
        assert(np.all(rcoords == coords))
        assert(np.all(rdata_a == data))
        #assert(np.all(rdata_b == data))
        breakpoint()


    uri = write()
    read(uri)

test_write_sparse()