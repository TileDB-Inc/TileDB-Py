
#%%
import tiledb
import numpy as np
from tiledb.tests.common import assert_subarrays_equal

array_name = "variable_length_array"

#%%

def create_array():
    ctx = tiledb.Ctx()

    dom = tiledb.Domain(tiledb.Dim(name="rows", domain=(1, 4), tile=4, dtype=np.int64),
                        tiledb.Dim(name="cols", domain=(1, 4), tile=4, dtype=np.int64),
                        ctx=ctx)

    attrs = [
        tiledb.Attr(name="a1", var=True, dtype='U', ctx=ctx),
        tiledb.Attr(name="a2", var=True, dtype=np.int64, ctx=ctx)
        ]

    schema = tiledb.ArraySchema(domain=dom, sparse=False,
                                attrs=attrs,
                                ctx=ctx)

    tiledb.Array.create(array_name, schema, ctx=ctx)

    return schema


def generate_data():
    # generate test input data
    a1_data = np.array([
                    "a", "bb", "ccc", "dd",
                    "eee", "f", "g", "hhh",
                    "i", "jjj", "kk", "l",
                    "m", "n", "oo", "p"
                ], dtype=np.object)

    a1_data = a1_data.reshape(4,4)

    a2_data = np.array(
                list(map(
                    lambda v: np.repeat(v[0], v[1]).astype(np.int64),
                    [
                    (1,1), (2,2), (3,1), (4,1),
                    (5,1), (6,2), (7,2), (8,3),
                    (9,2), (10,1),(11,1),(12,2),
                    (13,1),(14,3),(15,1),(16,1),
                    ]
                )), dtype=np.object)
    a2_data = a2_data.reshape(4,4)

    data_dict = { 'a1': a1_data,
                  'a2': a2_data
                }

    return data_dict


def write_array(data_dict):
    ctx = tiledb.Ctx()

    # open array for writing, and write data
    with tiledb.open(array_name, 'w', ctx=ctx) as array:
        array[:] = data_dict

def test_output_subarrays(test_dict):
    from numpy.testing import assert_array_equal

    ctx = tiledb.Ctx()
    with tiledb.open(array_name, ctx=ctx) as A:
        rt_dict = A[:]
        assert_subarrays_equal(test_dict['a2'], rt_dict['a2'])

create_array()
data = generate_data()
write_array(data)
test_output_subarrays(data)
