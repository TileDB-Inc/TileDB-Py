import numpy as np
import sys
import tiledb

array_name = "quickstart_kv"


def create_array():
    # Create a TileDB context
    ctx = tiledb.Ctx()

    # Check if the array already exists.
    if tiledb.object_type(ctx, array_name) == "kv":
        print("KV Array already exists.")
        sys.exit(0)

    # The KV array will have a single attribute "a" storing an integer.
    schema = tiledb.KVSchema(ctx, attrs=[tiledb.Attr(ctx, name="a", dtype=bytes)])

    # Create the (empty) array on disk.
    tiledb.KV.create(ctx, array_name, schema)


def write_array():
    ctx = tiledb.Ctx()
    # Open the array and write to it.
    A = tiledb.KV(ctx, array_name)
    A["key_1"] = "1"
    A["key_2"] = "2"
    A["key_3"] = "3"
    A.flush()


def read_array():
    ctx = tiledb.Ctx()
    # Open the array and read from it.
    A = tiledb.KV(ctx, array_name)
    print("key_1: %s" % A["key_1"])
    print("key_2: %s" % A["key_2"])
    print("key_3: %s" % A["key_3"])


create_array()
write_array()
read_array()
