# quickstart_dimension_labels_sparse.py
#
# LICENSE
#
# The MIT License
#
# Copyright (c) 2023 TileDB, Inc.
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
# Please refer to the TileDB and TileDB-Py documentation for more information:
#   https://docs.tiledb.com/main/how-to
#   https://tiledb-inc-tiledb.readthedocs-hosted.com/projects/tiledb-py/en/stable/python-api.html
#
# When run, this program will create a simple 2D sparse array with dimension labels, write
# some data to it, and read a slice of the data back.


import numpy as np

import tiledb


def create_array(uri: str):
    """Create array schema with dimension labels"""
    dim1 = tiledb.Dim("index1", domain=(0, 5))
    dim2 = tiledb.Dim("index2", domain=(0, 5))
    dom = tiledb.Domain(dim1, dim2)
    att = tiledb.Attr("values", dtype=np.int64)
    dim_labels = {
        0: {"x1": dim1.create_label_schema("increasing", np.float64, tile=6)},
        1: {"x2": dim2.create_label_schema("increasing", np.float64, tile=6)},
    }
    schema = tiledb.ArraySchema(
        domain=dom, attrs=(att,), dim_labels=dim_labels, sparse=True
    )
    tiledb.Array.create(uri, schema)


def write_array(uri: str):
    """Write attribute and label data to the array"""
    # Use dimension label URIs to write directly to dimension label arrays.
    schema = tiledb.ArraySchema.load(uri)
    with tiledb.open(schema.dim_label("x1").uri, "w") as x1_array:
        x1_array[:] = np.linspace(-1.0, 1.0, 6)
    with tiledb.open(schema.dim_label("x2").uri, "w") as x2_array:
        x2_array[:] = np.linspace(-1.0, 1.0, 6)

    # Write a tridiagonal matrix.
    coords1 = np.append(np.arange(1, 6), np.arange(5))
    coords2 = np.append(np.arange(5), np.arange(1, 6))
    data = np.append(np.arange(1, 6), -np.arange(1, 6))
    with tiledb.open(uri, "w") as array:
        array[coords1, coords2] = data


def read_array(uri: str):
    """Read the array from the dimension label"""

    with tiledb.open(uri, "r") as array:
        # Create a subarray and set label ranges.
        subarray = tiledb.Subarray(array)
        subarray.add_label_range("x1", (0.0, 1.0))
        subarray.add_label_range("x2", (0.0, 1.0))

        # Get data from the main array.
        data = array.read_subarray(subarray)

        # Read the labels.
        x1_label = array.schema.dim_label("x1")
        x2_label = array.schema.dim_label("x2")
        with tiledb.open(x1_label.uri) as x1_array:
            x1_coords = x1_array.multi_index[data["index1"]]
            data["x1"] = x1_coords[x1_label.label_attr_name]
        with tiledb.open(array.schema.dim_label("x2").uri) as x2_array:
            x2_coords = x2_array.multi_index[data["index2"]]
            data["x2"] = x2_coords[x2_label.label_attr_name]

        print("Reading data where 0.0 <= x1 <= 2.0 and -2.0 <= x2 <= 0.0")
        for name, value in data.items():
            print(f"  '{name}'={value}")


if __name__ == "__main__":
    # Name of the array to create.
    ARRAY_NAME = "quickstart_labels_sparse"

    LIBVERSION = tiledb.libtiledb.version()

    if LIBVERSION[0] == 2 and LIBVERSION[1] < 15:
        print(
            f"Dimension labels requires libtiledb version >= 2.15.0. Current version is"
            f" {LIBVERSION[0]}.{LIBVERSION[1]}.{LIBVERSION[2]}"
        )

    else:
        # Only create and write to the array if it doesn't already exist.
        if tiledb.object_type(ARRAY_NAME) != "array":
            create_array(ARRAY_NAME)
            write_array(ARRAY_NAME)

        # Read from the array and print output.
        read_array(ARRAY_NAME)
