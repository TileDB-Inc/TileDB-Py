# incomplete_iteration.py
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
#   https://docs.tiledb.com/main/how-to/arrays/reading-arrays/incomplete-queries
#
# When run, this program will create a 1D dense array, write some data
# to it, and read slices back by iteration over incomplete queries.
#


import numpy as np

import tiledb


def check_dataframe_deps():
    pd_error = """Pandas version >= 1.0 and < 3.0 required for dataframe functionality.
                  Please `pip install pandas>=1.0,<3.0` to proceed."""

    try:
        import pandas as pd
    except ImportError:
        raise Exception(pd_error)

    from packaging.version import Version

    if Version(pd.__version__) < Version("1.0") or Version(pd.__version__) >= Version(
        "3.0.0.dev0"
    ):
        raise Exception(pd_error)


# Name of the array to create.
array_name = "incomplete_iteration"


def create_array():
    # The array will be 100 cells with dimensions "x".
    dom = tiledb.Domain(tiledb.Dim(name="x", domain=(0, 99), tile=100, dtype=np.int64))

    # The array will be dense with a single string typed attribute "a"
    schema = tiledb.ArraySchema(
        domain=dom, sparse=True, attrs=[tiledb.Attr(name="a", dtype=str)]
    )

    # Create the (empty) array on disk.
    tiledb.SparseArray.create(array_name, schema)


def write_array():
    # Open the array and write to it.
    with tiledb.open(array_name, mode="w") as A:
        extent = A.schema.domain.dim("x").domain
        ncells = extent[1] - extent[0] + 1

        # Data is the Latin alphabet with varying repeat lengths
        data = [chr(i % 26 + 97) * (i % 52) for i in range(ncells)]

        # Coords are the dimension range
        coords = np.arange(extent[0], extent[1] + 1)

        A[coords] = data


def read_array_iterated():
    # in order to force iteration, restrict the buffer sizes
    # this setting gives 5 iterations for the example data
    init_buffer_bytes = 800
    cfg = tiledb.Config(
        {
            "py.init_buffer_bytes": init_buffer_bytes,
            "py.exact_init_buffer_bytes": "true",
        }
    )

    with tiledb.open(array_name, config=cfg) as A:
        # iterate over results as a dataframe
        iterable = A.query(return_incomplete=True).df[:]

        for i, result in enumerate(iterable):
            print(f"--- result {i} is a '{type(result)}' with size {len(result)}")
            print(result)
            print("---")

    print(f"Query completed after {i} iterations")


check_dataframe_deps()
create_array()
write_array()
read_array_iterated()
