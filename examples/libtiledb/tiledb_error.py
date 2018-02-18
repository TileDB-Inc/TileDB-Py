#/usr/bin/python

"""
This example shows how to catch errors

Simply run:

    $ python tiledb_error.py
"""

import tiledb


def main():
    ctx = tiledb.Ctx()
    try:
        tiledb.group_create(ctx, "mygroup")
        tiledb.group_create(ctx, "mygroup")
    except tiledb.TileDBError as err:
        print("TileDB exception: {!r}".format(err))


if __name__ == '__main__':
    main()
