# fragment_info.py
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

import numpy as np
import sys
import tiledb

array_name = "fragment_info"


def create_array():
    # The array will be 4x4 with dimensions "rows" and "cols", with domain [1,4] and space tiles 2x2.
    dom = tiledb.Domain(
        tiledb.Dim(name="rows", domain=(1, 4), tile=2, dtype=np.int32),
        tiledb.Dim(name="cols", domain=(1, 4), tile=2, dtype=np.int32),
    )

    # The array will be dense with a single attribute "a" so each (i,j) cell can store an integer.
    schema = tiledb.ArraySchema(
        domain=dom, sparse=False, attrs=[tiledb.Attr(name="a", dtype=np.int32)]
    )

    # Create the (empty) array on disk.
    tiledb.DenseArray.create(array_name, schema)


def write_array_1():
    with tiledb.DenseArray(array_name, mode="w") as A:
        A[1:3, 1:5] = np.array(([1, 2, 3, 4, 5, 6, 7, 8]))


def write_array_2():
    with tiledb.DenseArray(array_name, mode="w") as A:
        A[2:4, 2:4] = np.array(([101, 102, 103, 104]))


def write_array_3():
    with tiledb.DenseArray(array_name, mode="w") as A:
        A[3:4, 4:5] = np.array(([202]))


# Create and write array only if it does not exist
if tiledb.object_type(array_name) != "array":
    create_array()
    write_array_1()
    write_array_2()
    write_array_3()

fi = tiledb.fragment_info(array_name)

# Note that load() needs to be called each time the array is written to in
# order to get updated fragment information.
fi.load()

schema = tiledb.ArraySchema.load(array_name)

for fragment_num in range(fi.fragment_num()):
    print("Fragment number: {}".format(fragment_num))
    print(
        "\t> Non-Empty Domain: {}".format(fi.get_non_empty_domain(schema, fragment_num))
    )
    print("\t> URI: {}".format(fi.fragment_uri(fragment_num)))
    print("\t> Fragment Version: {}".format(fi.version(fragment_num)))
    print("\t> Timestamp Range: {}".format(fi.timestamp_range(fragment_num)))
    print("\t> Is DenseArray: {}".format(fi.dense(fragment_num)))
    print("\t> Is SparseArray: {}".format(fi.sparse(fragment_num)))
    print(
        "\t> Has Consolidated Metadata: {}".format(
            fi.has_consolidated_metadata(fragment_num)
        )
    )
    print()
