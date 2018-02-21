#!/usr/bin/python

"""
This example shows how to a create a KV object, read / write / update, and convert it
to a Python dict after update consolidation.

Simply run:

    $ python tiledb_kv.py
"""

import tiledb


def main():
    # Create TileDB context
    ctx = tiledb.Ctx()

    # KV objects are limited to storing string keys/values for the time being
    a1 = tiledb.Attr(ctx, "value", compressor=("gzip", -1), dtype=bytes)
    kv = tiledb.KV(ctx, "my_kv", attrs=(a1,))

    # Dump the KV schema
    kv.dump()

    # Update the KV with some key-value pairs
    vals = {"key1": "a", "key2": "bb", "key3": "dddd"}
    print("Updating KV with values: {!r}\n".format(vals))
    kv.update(vals)

    # Get kv item
    print("KV value for 'key3': {}\n".format(kv['key3']))

    try:
        kv["don't exist"]
    except KeyError:
        print("KeyError was raised for key 'don't exist'\n")

    # Set kv item
    kv['key3'] = "eeeee"
    print("Updated KV value for 'key3': {}\n".format(kv['key3']))

    # Consolidate kv updates
    kv.consolidate()

    # Convert kv to Python dict
    kv_dict = dict(kv)
    print("Convert to Python dict: {!r}\n".format(kv_dict))


if __name__ == '__main__':
    main()
