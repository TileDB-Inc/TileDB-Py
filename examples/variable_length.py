# variable_length.py
#
# LICENSE
#
# The MIT License
#
# Copyright (c) 2020 TileDB, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# DESCRIPTION
#
# Please see the TileDB documentation for more information:
#    https://docs.tiledb.com/main/solutions/tiledb-embedded/api-usage/writing-arrays/var-length-attributes
#
# This program shows how to set/get the TileDB configuration parameters.
#


#%%
import tiledb
import numpy as np
from tiledb.tests.common import assert_subarrays_equal

array_name = "variable_length_array"

#%%


def create_array():
    dom = tiledb.Domain(
        tiledb.Dim(name="rows", domain=(1, 4), tile=4, dtype=np.int64),
        tiledb.Dim(name="cols", domain=(1, 4), tile=4, dtype=np.int64),
    )

    attrs = [
        tiledb.Attr(name="a1", var=True, dtype="U"),
        tiledb.Attr(name="a2", var=True, dtype=np.int64),
    ]

    schema = tiledb.ArraySchema(domain=dom, sparse=False, attrs=attrs)

    tiledb.Array.create(array_name, schema)

    return schema


def generate_data():
    # generate test input data
    a1_data = np.array(
        [
            "a",
            "bb",
            "ccc",
            "dd",
            "eee",
            "f",
            "g",
            "hhh",
            "i",
            "jjj",
            "kk",
            "l",
            "m",
            "n",
            "oo",
            "p",
        ],
        dtype=np.object,
    )

    a1_data = a1_data.reshape(4, 4)

    a2_data = np.array(
        list(
            map(
                lambda v: np.repeat(v[0], v[1]).astype(np.int64),
                [
                    (1, 1),
                    (2, 2),
                    (3, 1),
                    (4, 1),
                    (5, 1),
                    (6, 2),
                    (7, 2),
                    (8, 3),
                    (9, 2),
                    (10, 1),
                    (11, 1),
                    (12, 2),
                    (13, 1),
                    (14, 3),
                    (15, 1),
                    (16, 1),
                ],
            )
        ),
        dtype=np.object,
    )
    a2_data = a2_data.reshape(4, 4)

    data_dict = {"a1": a1_data, "a2": a2_data}

    return data_dict


def write_array(data_dict):
    # open array for writing, and write data
    with tiledb.open(array_name, "w") as array:
        array[:] = data_dict


def test_output_subarrays(test_dict):
    with tiledb.open(array_name) as A:
        rt_dict = A[:]
        assert_subarrays_equal(test_dict["a2"], rt_dict["a2"])


create_array()
data = generate_data()
write_array(data)
test_output_subarrays(data)
