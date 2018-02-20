#!/usr/bin/python

"""
This example how to construct a sparse array (schema)

Simply run:

    $ python tiledb_array_schema.py
"""

import tiledb


def main():

    ctx = tiledb.Ctx()

    # create dimensions
    d1 = tiledb.Dim(ctx, "", domain=(1, 1000), tile=10, dtype="uint64")
    d2 = tiledb.Dim(ctx, "d2", domain=(101, 10000), tile=100, dtype="uint64")

    # create domain
    domain = tiledb.Domain(ctx, d1, d2)

    # create attributes
    a1 = tiledb.Attr(ctx, "", dtype="int32,int32,int32")
    a2 = tiledb.Attr(ctx, "a2", compressor=("gzip", -1), dtype="float32")

    # create sparse array with schema
    schema = tiledb.SparseArray(ctx, "sparse_array_schema",
                                domain=domain, attrs=(a1, a2),
                                capacity=10,
                                tile_order='row-major',
                                cell_order='col-major',
                                coords_compressor=('zstd', 4),
                                offsets_compressor=('blosc-lz', 5))
    schema.dump()

    # Print from schema
    print("From schema properties:")
    print("- Array type: ", "sparse" if schema.sparse else "dense")
    print("- Cell order: ", schema.cell_order)
    print("- Tile order: ", schema.tile_order)
    print("- Capacity: ", schema.capacity)
    print("- Coordinates compressor: ", schema.coords_compressor)
    print("- Offsets compressor: ", schema.offsets_compressor)
    print()

    # Print the attribute names:
    print("Array schema attribute names: ")
    for i in range(schema.nattr):
        print("* {!r}".format(schema.attr(i).name))
    print()

    # Print domain
    domain = schema.domain
    domain.dump()

    # print the dimension names
    print("Array schema dimension names: ")
    for i in range(schema.ndim):
        dim = domain.dim(i)
        print("* {!r}".format(dim.name))
    print()


if __name__ == '__main__':
    main()