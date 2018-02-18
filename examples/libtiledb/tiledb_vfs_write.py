#!/usr/bin/python

"""
This example shows how to write with VFS.

Simply run:

    $ python tiledb_.py
"""

import tiledb
import struct


def main():
    ctx = tiledb.Ctx()
    vfs = tiledb.VFS(ctx)

    # write binary data
    fh = vfs.open("tiledb_vfs.bin", mode='w')
    f1, s1 = 153.0, b'abcd'
    vfs.write(fh, struct.pack("f", f1))
    vfs.write(fh, s1)
    vfs.close(fh)

    # write binary data again, this will overwrite the previous file
    fh = vfs.open("tiledb_vfs.bin", mode='w')
    f1, s2 = 153.1, b'abcdef'
    vfs.write(fh, struct.pack("f", f1))
    vfs.write(fh, s2)
    vfs.close(fh)

    # append binary data to existing file
    fh = vfs.open("tiledb_vfs.bin", mode='a')
    s3 = b'ghijkl'
    vfs.write(fh, s3)
    vfs.close(fh)


if __name__ == '__main__':
    main()