# kv.py
#
# LICENSE
#
# The MIT License
#
# Copyright (c) 2018 TileDB, Inc.
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
# This is a part of the TileDB tutorial:
#   https://docs.tiledb.io/en/latest/tutorials/kv.html
#
# This program shows the various ways you can use a TileDB key-value store.
#

import numpy as np
import tiledb

array_name = "kv_array"


def create_array():
    # Create a TileDB context
    ctx = tiledb.Ctx()

    # Create a KV with a single attribute.
    # Note: KV arrays with multiple attributes or non-string-valued attributes
    # are not currently supported in Python.
    schema = tiledb.KVSchema(ctx, attrs=[tiledb.Attr(ctx, name="a1", dtype=bytes)])

    # Create the (empty) array on disk.
    tiledb.KV.create(ctx, array_name, schema)


def write_array():
    ctx = tiledb.Ctx()
    # Open the array and write to it.
    # We can optionally set the number of items to buffer before a flush.
    with tiledb.KV(ctx, array_name, mode='w') as A:
        A["key_1"] = "1"
        A["key_2"] = "2"
        A["key_3"] = "3"
        A.flush()


def read_array():
    ctx = tiledb.Ctx()
    # Open the array and read from it.
    with tiledb.KV(ctx, array_name, mode='r') as A:
        print("key_1: %s" % A["key_1"])
        print("key_2: %s" % A["key_2"])
        print("key_3: %s" % A["key_3"])


def iter_kv():
    ctx = tiledb.Ctx()
    with tiledb.KV(ctx, array_name, mode='r') as A:
        for p in A:
            print("key: '%s', value: '%s'" % (p[0], p[1]))


if tiledb.object_type(tiledb.Ctx(), array_name) != "kv":
    create_array()
    write_array()

read_array()
iter_kv()
