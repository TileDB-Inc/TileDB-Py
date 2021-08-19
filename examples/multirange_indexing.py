# multirange_indexing.py
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
#  https://docs.tiledb.com/main/solutions/tiledb-embedded/api-usage/reading-arrays/multi-range-subarrays
#
# When run, this program will create a simple 2D dense array with two
# attributes, write some data to it, and read a slice of the data back on
# (i) both attributes, and (ii) subselecting on only one of the attributes.
#


import numpy as np
import sys
import tiledb

# Name of the array to create.
array_name = "multi_range"


def create_array():
    # Check if the array already exists.
    if tiledb.object_type(array_name) == "array":
        return

    dom = tiledb.Domain(
        tiledb.Dim(name="x", domain=(1, 20), tile=4, dtype=np.int64),
        tiledb.Dim(name="y", domain=(1, 20), tile=4, dtype=np.int64),
    )

    # Add a single "a" float attribute
    schema = tiledb.ArraySchema(
        domain=dom, sparse=False, attrs=[tiledb.Attr(name="a", dtype=np.float64)]
    )

    # Create the (empty) array on disk.
    tiledb.DenseArray.create(array_name, schema)


def write_array():
    # Open the array and write to it.
    with tiledb.DenseArray(array_name, mode="w") as A:
        data_a = np.arange(400).reshape(20, 20)
        A[:, :] = {"a": data_a}


def read_array():
    # Open the array and read from it.
    with tiledb.DenseArray(array_name, mode="r") as A:
        # Slice only rows: (1,3) inclusive, and 5
        #            cols: 2, 5, 7
        data = A.multi_index[[(1, 3), 5], [2, 5, 7]]
        print("Reading attribute 'a', [ [1:3, 5], [2,5,7] ]")
        a = data["a"]
        print(a)


create_array()
write_array()
read_array()
