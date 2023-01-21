# multi_attribute.py
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
#   https://docs.tiledb.com/main/how-to/arrays/reading-arrays/multi-range-subarrays
#
# When run, this program will create a simple 2D dense array with two
# attributes, write some data to it, and read a slice of the data back on
# (i) both attributes, and (ii) subselecting on only one of the attributes.
#


import numpy as np

import tiledb

# Name of the array to create.
array_name = "multi_attribute"


def create_array():
    # Check if the array already exists.
    if tiledb.object_type(array_name) == "array":
        return

    # The array will be 4x4 with dimensions "rows" and "cols", with domain [1,4].
    dom = tiledb.Domain(
        tiledb.Dim(name="rows", domain=(1, 4), tile=4, dtype=np.int32),
        tiledb.Dim(name="cols", domain=(1, 4), tile=4, dtype=np.int32),
    )

    # Add two attributes "a1" and "a2", so each (i,j) cell can store
    # a character on "a1" and a vector of two floats on "a2".
    schema = tiledb.ArraySchema(
        domain=dom,
        sparse=False,
        attrs=[
            tiledb.Attr(name="a1", dtype=np.uint8),
            tiledb.Attr(
                name="a2",
                dtype=np.dtype([("", np.float32), ("", np.float32), ("", np.float32)]),
            ),
        ],
    )

    # Create the (empty) array on disk.
    tiledb.DenseArray.create(array_name, schema)


def write_array():
    # Open the array and write to it.
    with tiledb.DenseArray(array_name, mode="w") as A:
        data_a1 = np.array(
            (
                list(
                    map(
                        ord,
                        [
                            "a",
                            "b",
                            "c",
                            "d",
                            "e",
                            "f",
                            "g",
                            "h",
                            "i",
                            "j",
                            "k",
                            "l",
                            "m",
                            "n",
                            "o",
                            "p",
                        ],
                    )
                )
            )
        )
        data_a2 = np.array(
            (
                [
                    (1.1, 1.2, 1.3),
                    (2.1, 2.2, 2.3),
                    (3.1, 3.2, 3.3),
                    (4.1, 4.2, 4.3),
                    (5.1, 5.2, 5.3),
                    (6.1, 6.2, 6.3),
                    (7.1, 7.2, 7.3),
                    (8.1, 8.2, 8.3),
                    (9.1, 9.2, 9.3),
                    (10.1, 10.2, 10.3),
                    (11.1, 11.2, 11.3),
                    (12.1, 12.2, 12.3),
                    (13.1, 13.2, 13.3),
                    (14.1, 14.2, 14.3),
                    (15.1, 15.2, 15.3),
                    (16.1, 16.2, 16.3),
                ]
            ),
            dtype=[("", np.float32), ("", np.float32), ("", np.float32)],
        )
        A[:, :] = {"a1": data_a1, "a2": data_a2}


def read_array():
    # Open the array and read from it.
    with tiledb.DenseArray(array_name, mode="r") as A:
        # Slice only rows 1, 2 and cols 2, 3, 4.
        data = A[1:3, 2:5]
        print("Reading both attributes a1 and a2:")
        a1, a2 = data["a1"].flat, data["a2"].flat
        for i, v in enumerate(a1):
            print(
                "a1: '%s', a2: (%.1f,%.1f,%.1f)"
                % (chr(v), a2[i][0], a2[i][1], a2[i][2])
            )


def read_array_subselect():
    # Open the array and read from it.
    with tiledb.DenseArray(array_name, mode="r") as A:
        # Slice only rows 1, 2 and cols 2, 3, 4, attribute 'a1' only.
        # We use the '.query()' syntax which allows attribute subselection.
        data = A.query(attrs=["a1"])[1:3, 2:5]
        print("Subselecting on attribute a1:")
        for a in data["a1"].flat:
            print("a1: '%s'" % chr(a))


create_array()
write_array()
read_array()
read_array_subselect()
