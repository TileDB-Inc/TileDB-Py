# object.py
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
#   https://docs.tiledb.io/en/latest/tutorials/object.html
#
# This program creates a hierarchy as shown below. Specifically, it creates
# groups `dense_arrays` and `sparse_arrays` in a group `my_group`, and
# then some dense/sparse arrays and key-value store in those groups.
#
# my_group/
#   - dense_arrays/
#     - array_A
#     - array_B
#     - kv
#   - sparse_arrays/
#     - array_C
#     - array_D
#
# The program then shows how to list this hierarchy, as well as
# move/remove TileDB objects.
#

# For print() in Python 2
from __future__ import print_function

import numpy as np
import tiledb


def create_array(array_name, sparse):
    ctx = tiledb.Ctx()
    if tiledb.object_type(ctx, array_name) == "array":
        return

    dom = tiledb.Domain(ctx,
                        tiledb.Dim(ctx, name="rows", domain=(1, 4), tile=4, dtype=np.int32),
                        tiledb.Dim(ctx, name="cols", domain=(1, 4), tile=4, dtype=np.int32))
    schema = tiledb.ArraySchema(ctx, domain=dom, sparse=sparse,
                                attrs=[tiledb.Attr(ctx, name="a", dtype=np.int32)])
    if sparse:
        tiledb.SparseArray.create(array_name, schema)
    else:
        tiledb.DenseArray.create(array_name, schema)


def create_kv(array_name):
    ctx = tiledb.Ctx()
    schema = tiledb.KVSchema(ctx, attrs=[tiledb.Attr(ctx, name="a", dtype=bytes)])
    tiledb.KV.create(ctx, array_name, schema)


def create_hierarchy():
    ctx = tiledb.Ctx()

    # Create groups
    tiledb.group_create(ctx, "my_group")
    tiledb.group_create(ctx, "my_group/dense_arrays")
    tiledb.group_create(ctx, "my_group/sparse_arrays")

    # Create arrays
    create_array("my_group/dense_arrays/array_A", False)
    create_array("my_group/dense_arrays/array_B", False)
    create_array("my_group/sparse_arrays/array_C", True)
    create_array("my_group/sparse_arrays/array_D", True)

    # Create key-value store
    create_kv("my_group/dense_arrays/kv")


def list_obj(path):
    ctx = tiledb.Ctx()

    # List children
    print("\nListing hierarchy:")
    tiledb.ls(ctx, path, lambda obj_path, obj_type: print(obj_path, obj_type))

    # Walk in a path with a pre- and post-order traversal
    print("\nPreorder traversal:")
    tiledb.walk(ctx, path, lambda obj_path, obj_type: print(obj_path, obj_type))  # Default order is preorder

    print("\nPostorder traversal:")
    tiledb.walk(ctx, path, lambda obj_path, obj_type: print(obj_path, obj_type), "postorder")


def move_remove_obj():
    ctx = tiledb.Ctx()
    tiledb.move(ctx, "my_group", "my_group_2")
    tiledb.remove(ctx, "my_group_2/dense_arrays")
    tiledb.remove(ctx, "my_group_2/sparse_arrays/array_C")


create_hierarchy()
list_obj("my_group")
move_remove_obj()  # Renames 'my_group' to 'my_group_2'
list_obj("my_group_2")
