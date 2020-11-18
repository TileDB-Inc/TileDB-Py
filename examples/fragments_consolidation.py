# fragments_consolidation.py
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
#    https://docs.tiledb.com/main/solutions/tiledb-embedded/api-usage/consolidation-and-vacuuming
#
# When run, this program will create a simple 2D dense array, write some data
# with three queries (creating three fragments), optionally consolidate
# and read the entire array data back.
#

import numpy as np
import sys
import tiledb

array_name = "fragments_consolidation"


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
        # Note: sparse (unordered) writes to dense arrays are not yet supported in Python.
        # Instead we can make two single-cell writes (results in total of 4 fragments).
        A[1:2, 1:2] = np.array(([201]))
        A[3:4, 4:5] = np.array(([202]))


def read_array():
    with tiledb.DenseArray(array_name, mode="r") as A:
        # Read the entire array. To get coord values as well, we use the .query() syntax.
        data = A.query(coords=True)[:, :]
        a_vals = data["a"]
        rows = data["rows"]
        cols = data["cols"]
        for i in range(rows.shape[0]):
            for j in range(cols.shape[0]):
                print(
                    "Cell {} has data {}".format(
                        str((rows[i, j], cols[i, j])), str(a_vals[i, j])
                    )
                )


# Create and write array only if it does not exist
if tiledb.object_type(array_name) != "array":
    create_array()
    write_array_1()
    write_array_2()
    write_array_3()

# Optionally consolidate
if len(sys.argv) > 1 and sys.argv[1] == "consolidate":
    config = tiledb.Config()
    config["sm.consolidation.steps"] = 1
    config["sm.consolidation.step_max_frags"] = 3
    config["sm.consolidation.step_min_frags"] = 1
    tiledb.consolidate(config=config, uri=array_name)

read_array()
