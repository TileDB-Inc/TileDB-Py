# quickstart_dense.py
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
#  https://docs.tiledb.com/main/how-to/arrays/creating-arrays/creating-dimensions
#
# When run, this program will create a simple 2D dense array, write some data
# to it, and read a slice of the data back.
#


import numpy as np

import tiledb

path = "sparse_mixed_demo"

dom = tiledb.Domain(
    *[
        tiledb.Dim(name="str_dim", domain=(None, None), dtype=np.bytes_),
        tiledb.Dim(name="int64_dim", domain=(0, 100), tile=10, dtype=np.int64),
        tiledb.Dim(
            name="float64_dim",
            domain=(-100.0, 100.0),
            tile=10,
            dtype=np.float64,
        ),
    ],
)

att = tiledb.Attr(name="a", dtype=np.int64)
schema = tiledb.ArraySchema(domain=dom, attrs=(att,), sparse=True, capacity=10000)
tiledb.SparseArray.create(path, schema)

data = [1, 2, 3, 4]
c_str = [b"aa", b"bbb", b"c", b"dddd"]
c_int64 = [0, 10, 20, 30]
c_float64 = [-95.0, -61.5, 1.3, 42.7]

with tiledb.open(path, "w") as A:
    A[c_str, c_int64, c_float64] = data

with tiledb.open(path) as A:
    print("\n\nRead full array:\n")
    print(A[:])

    print("\n\nRead string slice A['c':'dddd']:\n")
    print(A["c":"dddd"])

    print("\n\nRead A[:, 10]: \n")
    print(A["aa":"bbb"])

    print("\n\nRead A.multi_index['aa':'c', 0:10]\n")
    print(A.multi_index["aa":"c", 0:10])

    print("\n\nRead A.multi_index['aa':'bbb', :, -95.0:-61.5]\n")
    print(A.multi_index["aa":"bbb", :, -95.0:-61.5])
