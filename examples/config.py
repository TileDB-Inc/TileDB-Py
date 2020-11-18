# config.py
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
#    https://docs.tiledb.com/main/solutions/tiledb-embedded/api-usage/configuration
#
# This program shows how to set/get the TileDB configuration parameters.
#

import tiledb


def set_get_config_ctx_vfs():
    # Create config object
    config = tiledb.Config()

    # Set/get config to/from ctx
    ctx = tiledb.Ctx(config)
    config_ctx = ctx.config()

    # Set/get config to/from VFS
    vfs = tiledb.VFS(config)
    config_vfs = vfs.config()


def set_get_config():
    config = tiledb.Config()

    # Set value
    config["vfs.s3.connect_timeout_ms"] = 5000

    # Get value
    tile_cache_size = config["sm.tile_cache_size"]
    print("Tile cache size: %s" % str(tile_cache_size))


def print_default():
    config = tiledb.Config()
    print("\nDefault settings:")
    for p in config.items():
        print('"%s" : "%s"' % (p[0], p[1]))


def iter_config_with_prefix():
    config = tiledb.Config()
    # Print only the S3 settings.
    print("\nVFS S3 settings:")
    for p in config.items("vfs.s3."):
        print('"%s" : "%s"' % (p[0], p[1]))


def save_load_config():
    # Save to file
    config = tiledb.Config()
    config["sm.tile_cache_size"] = 0
    config.save("tiledb_config.txt")

    # Load from file
    config_load = tiledb.Config.load("tiledb_config.txt")
    print(
        "\nTile cache size after loading from file: %s"
        % str(config_load["sm.tile_cache_size"])
    )


set_get_config_ctx_vfs()
set_get_config()
print_default()
iter_config_with_prefix()
save_load_config()
