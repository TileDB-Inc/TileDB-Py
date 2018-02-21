#!/usr/bin/python

"""
This example shows how to create a dense array.
Make sure that no directory exists with the name
`my_dense_array` in the current working directory

Simply run:

    $ python tiledb_dense_create.py
"""

import tiledb


def main():
    ctx = tiledb.Ctx()

    # Create dimensions
    d1 = tiledb.Dim(ctx, "d1", domain=(1, 4), tile=2, dtype="uint64")
    d2 = tiledb.Dim(ctx, "d2", domain=(1, 4), tile=2, dtype="uint64")

    # Create domain
    domain = tiledb.Domain(ctx, d1, d2)

    # Create attributes
    a1 = tiledb.Attr(ctx, "a1", compressor=('blosc-lz', -1), dtype="int32")
    a2 = tiledb.Attr(ctx, "a2", compressor=("gzip", -1), dtype="S4")
    a3 = tiledb.Attr(ctx, "a3", compressor=('zstd', -1), dtype='float32,float32')

    # Create dense array
    tiledb.DenseArray(ctx, "my_dense_array",
                      domain=domain,
                      attrs=(a1, a2, a3),
                      cell_order='row-major',
                      tile_order='row-major')


if __name__ == '__main__':
    main()