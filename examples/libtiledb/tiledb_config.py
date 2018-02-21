#!/usr/bin/python

"""
This example shows how to manipulate Config parameters

Simply run:

    $ python tiledb_config.py
"""

import tiledb


def main():
    config = tiledb.Config()

    # Print the default config parameters
    print("Default settings:\n{0!r}".format(config))

    # Set values
    config["vfs.s3.connect_timeout_ms"] = 5000
    config["vfs.s3.endpoint_override"] = "localhost:88880"

    # Get values
    tile_cache_size = config["sm.tile_cache_size"]
    print("\nTile cache size: ", tile_cache_size)

    # Print only the s3 settings
    print("\nVFS S3 settings:")
    for p, v in config.items(prefix="vfs.s3."):
        print("{0!r} : {1!r}".format(p, v))

    # Assign a config object to Ctx and VFS
    ctx = tiledb.Ctx(config=config)
    vfs = tiledb.VFS(ctx, config=config)


if __name__ == '__main__':
    main()