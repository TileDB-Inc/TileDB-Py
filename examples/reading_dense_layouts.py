# reading_dense_layouts.py
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
#   https://docs.tiledb.com/main/solutions/tiledb-embedded/api-usage/reading-arrays/basic-reading
#
# When run, this program will create a simple 2D dense array, write some data
# to it, and read a slice of the data back in the layout of the user's choice
# (passed as an argument to the program: "row", "col", or "global").
#

import numpy as np
import sys
import tiledb

# Name of the array to create.
array_name = "reading_dense_layouts"


def create_array():
    # The array will be 4x4 with dimensions "rows" and "cols", with domain [1,4].
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


def write_array():
    # Open the array and write to it.
    with tiledb.DenseArray(array_name, mode="w") as A:
        # NOTE: global writes are not currently supported in the Python API.
        # The following code will produce the same array as the corresponding
        # C++ example in the docs (which wrote in global order)
        data = np.array(([1, 2, 5, 6], [3, 4, 7, 8], [9, 10, 13, 14], [11, 12, 15, 16]))
        A[:] = data


def read_array(order):
    # Open the array and read from it.
    with tiledb.DenseArray(array_name, mode="r") as A:
        # Get non-empty domain
        print("Non-empty domain: {}".format(A.nonempty_domain()))

        # Slice only rows 1, 2 and cols 2, 3, 4.
        # NOTE: The `query` syntax is required to get the coordinates for
        # dense arrays and specify an order other than the default row-major
        data = A.query(attrs=["a"], order=order, coords=True)[1:3, 2:5]
        a_vals = data["a"]
        coords = np.asarray(list(zip(data["rows"], data["cols"])))

        if order != "G" and a_vals.flags["F_CONTIGUOUS"]:
            print("NOTE: The following result array has col-major layout internally")

        if order != "G":
            for i in range(coords.shape[0]):
                for j in range(coords.shape[1]):
                    print(
                        "Cell {} has data {}".format(
                            str(coords[i, j]), str(a_vals[i, j])
                        )
                    )
        else:
            # When reading in global order, TileDB always returns a vector (1D array)
            for i in range(coords.shape[0]):
                print("Cell {} has data {}".format(str(coords[i]), str(a_vals[i])))


# Check if the array already exists.
if tiledb.object_type(array_name) != "array":
    create_array()
    write_array()

layout = ""
if len(sys.argv) > 1:
    layout = sys.argv[1]

order = "C"
if layout == "col":
    order = "F"
elif layout == "global":
    order = "G"
else:
    order = "C"

read_array(order)
