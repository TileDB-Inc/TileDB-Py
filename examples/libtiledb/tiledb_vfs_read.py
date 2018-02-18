#!/usr/bin/python

"""
Read from a file with VFS

Simply run:

    $ python tiledb_vfs_write.py
    $ python tiledb_vfs_read.py
"""

import tiledb
import struct


def main():
    # create TileDB context
    ctx = tiledb.Ctx()

    # create TileDB VFS
    vfs = tiledb.VFS(ctx)

    # Read binary data
    fh = vfs.open("tiledb_vfs.bin", mode='r')

    nbytes = struct.calcsize("f")
    f1 = vfs.read(fh, 0, nbytes)
    s1 = vfs.read(fh, nbytes, 12)

    print("Binary read:\n{0!r}\n{1!r}".format(struct.unpack("f", f1)[0], s1))

    vfs.close(fh)


if __name__ == '__main__':
    main()