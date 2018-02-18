#!/usr/bin/python

"""
List / Walk a directory for TileDB objects

Simply run:

    $ python tiledb_group_create.py
    $ python tiledb_object_ls_walk.py
"""

import tiledb


def main():
    ctx = tiledb.Ctx()

    def callback(obj_path, obj_type):
        print("{0!r} {1!r}".format(obj_path, obj_type))

    print("List children: ")
    tiledb.ls(ctx, "my_group", callback)

    print("\nPreorder traversal: ")
    tiledb.walk(ctx, "my_group", callback, order="preorder")

    print("\nPostorder traversal: ")
    tiledb.walk(ctx, "my_group", callback, order='postorder')


if __name__ == '__main__':
    main()