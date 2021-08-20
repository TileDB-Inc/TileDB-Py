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

# tiledb.array_fragments() requires TileDB-Py version > 0.8.5
fragments_info = tiledb.array_fragments(array_name)

print("====== FRAGMENTS  INFO ======")
print("array uri: {}".format(fragments_info.array_uri))
print("number of fragments: {}".format(len(fragments_info)))

to_vac = fragments_info.to_vacuum
print("number of consolidated fragments to vacuum: {}".format(len(to_vac)))
print("uris of consolidated fragments to vacuum: {}".format(to_vac))

print(fragments_info.nonempty_domain)
print(fragments_info.sparse)

for fragment in fragments_info:
    print()
    print("===== FRAGMENT NUMBER {} =====".format(fragment.num))
    print("fragment uri: {}".format(fragment.uri))
    print("is dense: {}".format(fragment.dense))
    print("is sparse: {}".format(fragment.sparse))
    print("cell num: {}".format(fragment.cell_num))
    print("has consolidated metadata: {}".format(fragment.has_consolidated_metadata))
    print("nonempty domain: {}".format(fragment.nonempty_domain))
    print("timestamp range: {}".format(fragment.timestamp_range))
    print(
        "number of unconsolidated metadata: {}".format(
            fragment.unconsolidated_metadata_num
        )
    )
    print("version: {}".format(fragment.version))
