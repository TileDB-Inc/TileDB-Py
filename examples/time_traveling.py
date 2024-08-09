# time_traveling.py
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
# When run, this program will create a simple sparse array, write some data
# to it at specified timestamps, and read the entire array data back.
#

import numpy as np

import tiledb

# Name of the array to create.
array_name = "time_traveling"


def create_array():
    dom = tiledb.Domain(tiledb.Dim(domain=(0, 0), tile=1, dtype=np.int64))
    att = tiledb.Attr(name="num", dtype=np.int64)
    schema = tiledb.ArraySchema(sparse=True, domain=dom, attrs=(att,))
    tiledb.SparseArray.create(array_name, schema)


def write_array():
    # Open the array and write to it.
    for timestamp in range(1, 4):
        with tiledb.open(array_name, timestamp=timestamp, mode="w") as T:
            T[0] = timestamp


def read_array():
    # Open the array and read from it.
    with tiledb.open(array_name, mode="r") as T:
        print(T[:]["num"])

    with tiledb.open(array_name, mode="r", timestamp=(1, 2)) as T:
        print(T[:]["num"])

    with tiledb.open(array_name, mode="r", timestamp=(2, 3)) as T:
        print(T[:]["num"])

    with tiledb.open(array_name, mode="r", timestamp=1) as T:
        print(T[:]["num"])

    with tiledb.open(array_name, mode="r", timestamp=(1, None)) as T:
        print(T[:]["num"])


if tiledb.object_type(array_name) != "array":
    create_array()
    write_array()

read_array()
