# global_order_writes.py
#
# LICENSE
#
# The MIT License
#
# Copyright (c) 2025 TileDB, Inc.
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
# This example demonstrates writing to TileDB arrays in global order using
# multiple submit() calls before finalize(). This is useful for writing large
# datasets in batches while ensuring only a single fragment is created.
#

import numpy as np

import tiledb

# Name of the arrays to create
sparse_array_name = "global_order_sparse"
dense_array_name = "global_order_dense"


def create_sparse_array():
    """Create a simple 1D sparse array."""
    dim = tiledb.Dim("d1", domain=(1, 1000), tile=100, dtype=np.int32)
    dom = tiledb.Domain(dim)
    att = tiledb.Attr("a1", dtype=np.int64)
    schema = tiledb.ArraySchema(domain=dom, attrs=(att,), sparse=True)
    tiledb.Array.create(sparse_array_name, schema)


def create_dense_array():
    """Create a simple 1D dense array."""
    dim = tiledb.Dim("d1", domain=(1, 1000), tile=100, dtype=np.int32)
    dom = tiledb.Domain(dim)
    att = tiledb.Attr("a1", dtype=np.int64)
    schema = tiledb.ArraySchema(domain=dom, attrs=(att,), sparse=False)
    tiledb.Array.create(dense_array_name, schema)


def write_sparse_global_order():
    """Write to sparse array in global order with multiple submits."""
    print("Writing sparse array in global order with multiple submits...")

    with tiledb.open(sparse_array_name, "w") as A:
        # Create a query with global order layout
        q = tiledb.Query(A, order="G")

        # First batch of data
        coords_batch1 = np.array([1, 5, 10, 20, 50], dtype=np.int32)
        data_batch1 = np.array([100, 200, 300, 400, 500], dtype=np.int64)
        q.set_data({"d1": coords_batch1, "a1": data_batch1})
        q.submit()

        # Second batch of data (coordinates must be in global order relative to first batch)
        coords_batch2 = np.array([100, 200, 500], dtype=np.int32)
        data_batch2 = np.array([600, 700, 800], dtype=np.int64)
        q.set_data({"d1": coords_batch2, "a1": data_batch2})
        q.submit()

        # Finalize to complete the write
        q.finalize()

    # Verify only one fragment was created
    fragments = tiledb.array_fragments(sparse_array_name)
    print(f"Number of fragments created: {len(fragments)}")
    print("Sparse array written successfully")


def write_dense_global_order():
    """Write to dense array in global order with multiple submits."""
    print("\nWriting dense array in global order with multiple submits...")

    with tiledb.open(dense_array_name, "w") as A:
        # Create a query with global order layout
        q = tiledb.Query(A, order="G")

        # Set the subarray to cover the full range we're writing
        start_coord = 1
        end_coord = 100
        q.set_subarray_ranges([(start_coord, end_coord)])

        # First batch of data (cells 1-50)
        data_batch1 = np.arange(1, 51, dtype=np.int64)
        q.set_data({"a1": data_batch1})
        q.submit()

        # Second batch of data (cells 51-100)
        data_batch2 = np.arange(51, 101, dtype=np.int64)
        q.set_data({"a1": data_batch2})
        q.submit()

        # Finalize to complete the write
        q.finalize()

    # Verify only one fragment was created
    fragments = tiledb.array_fragments(dense_array_name)
    print(f"Number of fragments created: {len(fragments)}")
    print("Dense array written successfully")


def read_and_verify():
    """Read back the data to verify it was written correctly."""
    print("\nReading and verifying data...")

    # Read sparse array
    with tiledb.open(sparse_array_name, "r") as A:
        result = A[:]
        print(f"Sparse array has {len(result['a1'])} cells")
        print(f"First 5 values: {result['a1'][:5]}")

    # Read dense array
    with tiledb.open(dense_array_name, "r") as A:
        result = A[1:20]  # Read cells 1-19
        print(f"Dense array cells 1-19: {result['a1'][:5]}...{result['a1'][-3:]}")


if __name__ == "__main__":
    print("=" * 60)
    print("TileDB Global Order Writes Example")
    print("=" * 60)

    # Check if arrays exist, create if not
    if tiledb.object_type(sparse_array_name) != "array":
        create_sparse_array()
    if tiledb.object_type(dense_array_name) != "array":
        create_dense_array()

    # Write data
    write_sparse_global_order()
    write_dense_global_order()

    # Read and verify
    read_and_verify()

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)
