#%%
import tiledb
#from tiledb.cppapi import cc as lt
import cc as lt

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.compute
import numpy as np
import hypothesis
import time
import tempfile
import os

from dataclasses import dataclass

from common import paths_equal

from typing import Dict, Tuple, Optional

import pytest

#%%
def get_data(uri):
    dts = ds.dataset(uri)

    assert len(dts.files) > 0

    return dts.to_table()

#%%
uri = "/Users/inorton/work/git/TileDB-Py-gen/cppapi/test_data/262205/"
table = get_data(uri)

#tiledb_uri = "/Users/inorton/work/git/TileDB-Py-gen/cppapi/test_data/ts_array.tdb"
tiledb_uri = "/Users/inorton/work/git/TileDB-Py-gen/cppapi/test_data/ts_array2.tdb/"
tiledb_array = tiledb.open(tiledb_uri)
#%%
def init_write(bath: pa.Table, array_uri: str):
    pass

@dataclass
class AttrMapper:
    _table_attr_map: Dict[str, str]
    _attr_field_map: Dict[str, str]

    def __init__(self, pa_schema, tdb_schema):
        _mattrn = lambda x: x.split(":", 1)[0]
        self._table_attr_map = {n: _mattrn(n) for n in pa_schema.names}
        self._attr_field_map = {_mattrn(n): n for n in pa_schema.names}

    def field2attr(self, field_name: str):
        return self._table_attr_map[field_name]

    def attr2field(self, attr_name: str):
        return self._attr_field_map[attr_name]

def convert_offsets(offsets):
    """
    Convert arrow offsets to tiledb offsets
    - remove last element
    - in-place multiplication by 8
    """
    res = offsets.view(np.uint32).astype(np.uint64)[:-1]
    np.multiply(res, 8) #, out=res)
    return res

def get_buffers(table: pa.Table, tdb_schema: tiledb.ArraySchema) -> Dict[str, Tuple[np.array]]:
    """
    This function will return a Dict[str, Tuple[np.array]] of tuples representing the buffers
    from the given pyarrow table.

    The tuple will have format:

        (data, Optional[offsets])
    """
    attr_mapper = AttrMapper(table.schema, tdb_schema)

    dims = dict()
    attrs = dict()

    for col_name in table.column_names:
        if col_name.startswith("imo"):
            continue
        attr_name = attr_mapper.field2attr(col_name)

        isdim = tdb_schema.domain.has_dim(attr_mapper.field2attr(col_name))
        #print(col_name, isdim)
        isvar = (not isdim) and tdb_schema.attr(attr_name).isvar

        col = table[col_name]
        #print(f"proc: ", col_name, f" isvar: {isvar}")

        if isvar:
            if isinstance(col, pa.ChunkedArray):
                assert isinstance(col.chunks[0], pa.StringArray), "Expected StringArrays for concat!"
                col = pa.concat_arrays(col.chunks)
            elif not isinstance(col, pa.StringArray):
                # if the array is not a StringArray, try to force type
                col = pa.array(col, pa.string())

            assert isinstance(col, pa.StringArray), f"'{col_name}' not a StringArray"
            buffers = col.buffers()
            res = (
                np.asarray(buffers[2]).view(np.uint8),
                convert_offsets(np.asarray(buffers[1]))
            )
        else:
            dtype = tdb_schema.domain.dim(attr_name).dtype if isdim else tdb_schema.attr(attr_name).dtype
            res = (
                # note that we must use dtype here to get proper conversion for datetime arrays
                np.asarray(col, dtype=dtype),
                None
            )

        if isdim:
            dims[attr_name] = res
        else:
            attrs[attr_name] = res

    return dims,attrs

    #buffers = pa_array.buffers()
    #return (
    #    np.asarray(
    #        buffers[0].

    #    )
    #)

#r = get_buffers(table, tiledb_array.schema)

#class ArrowWriter:
#    _dim_buffers: Dict[str, Tuple[np.array, Optional[np.array]]]
#    _attr_buffers: Dict[str, Tuple[np.array, Optional[np.array]]]
#
#    def __init__(self,)

def write_arrow_buffers(tiledb_uri: str, dim_buffers, attr_buffers):
    with tiledb.open(tiledb_uri) as tdb_r:
        tdb_schema = tdb_r.schema

    attr_buffers["imo"] = (np.arange(len(table)), None)

    get_name = lambda k: tdb_schema.domain.dim(k).name
    global dims
    dims = {get_name(k): dim_buffers[get_name(k)] for k in range(tdb_schema.ndim)}

    #config_dict = {
    #    "sm.var_offsets.bitsize": "32",
    #    "sm.var_offsets.mode": "elements",
    #    "sm.var_offsets.extra_element": "true"
    #}
    config_dict = {}

    config = lt.Config(config_dict)
    ctx = lt.Context(config)
    cc_array = lt.Array(ctx, tiledb_uri, lt.QueryType.WRITE)
    query = lt.Query(ctx, cc_array, lt.QueryType.WRITE)
    query.set_layout(lt.LayoutType.UNORDERED)

    for name, (data,offsets) in dims.items():
        query.set_data_buffer(name, data)
        if offsets is not None: # TBD
            query.set_offsets_buffer(name, offsets)

    for name, (data,offsets) in attr_buffers.items():
        query.set_data_buffer(name, data)
        if offsets is not None:
            query.set_offsets_buffer(name, offsets)


    query.submit()

    if not query.query_status() == lt.QueryStatus.COMPLETE:
        raise Exception("Query submit failed to complete!")

def write_arrow_table(table: pa.Table, tiledb_uri: str):
    with tiledb.open(tiledb_uri) as tdb_r:
        tdb_schema = tdb_r.schema

    global dim_buffers, attr_buffers
    dim_buffers, attr_buffers = get_buffers(table, tdb_schema)

    write_arrow_buffers(tiledb_uri, dim_buffers, attr_buffers)

#%%
#write_arrow_buffers(tiledb_uri, dim_buffers, attr_buffers)
# %%
write_arrow_table(table, tiledb_uri)

# %%
dim_buffers, attr_buffers = get_buffers(table, tiledb_array.schema)
# %%
