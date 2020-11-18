# using_tiledb_stats.py
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
#   https://docs.tiledb.com/main/solutions/tiledb-embedded/api-usage/using-performance-statistics
#
# When run, this program will create a 0.5GB dense array, and enable the
# TileDB statistics surrounding reads from the array.
#

import numpy as np
import tiledb

# Name of array.
array_name = "stats_array"


def create_array(row_tile_extent, col_tile_extent):
    dom = tiledb.Domain(
        tiledb.Dim(
            name="rows", domain=(1, 12000), tile=row_tile_extent, dtype=np.int32
        ),
        tiledb.Dim(
            name="cols", domain=(1, 12000), tile=col_tile_extent, dtype=np.int32
        ),
    )

    schema = tiledb.ArraySchema(
        domain=dom, sparse=False, attrs=[tiledb.Attr(name="a", dtype=np.int32)]
    )

    # Create the (empty) array on disk.
    tiledb.DenseArray.create(array_name, schema)


def write_array():
    # Open the array and write to it.
    with tiledb.DenseArray(array_name, mode="w") as A:
        data = np.arange(12000 * 12000)
        A[:] = data


def read_array():
    # Open the array and read from it.
    with tiledb.DenseArray(array_name, mode="r") as A:
        # Read a slice of 3,000 rows.
        # Enable the stats for the read query, and print the report.
        tiledb.stats_enable()
        data1 = A[1:3001, 1:12001]
        tiledb.stats_dump()
        tiledb.stats_disable()


# Create array with each row as a tile.
create_array(1, 12000)
write_array()
read_array()
