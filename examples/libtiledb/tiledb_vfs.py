#!/usr/bin/python

"""
Exploring VFS functionality

Simply run:

    $ python tiledb_.py
"""

import tiledb


def main():
    # Create TileDB context
    ctx = tiledb.Ctx()

    # Create TileDB VFS
    vfs = tiledb.VFS(ctx)

    # Create directory
    if not vfs.is_dir("dir_A"):
        vfs.create_dir("dir_A")
        print("Created dir_A")
    else:
        print("dir_A already exists")

    # Creating an(empty) file
    if not vfs.is_file("dir_A/file_A"):
        vfs.touch("dir_A/file_A")
        print("Created empty file dir_A/file_A")
    else:
        print("dir_A/file_A already exists")

    # Getting the file size
    print("File size: {0!s}".format(vfs.file_size("dir_A/file_A")))

    # Moving files(moving directories is similar)
    print("Moving file dir_A/file_A to dir_A/file_B")
    vfs.move("dir_A/file_A", "dir_A/file_B", force=True)

    # Deleting files and directories
    print("Deleting dir_A/file_B and dir_A")
    vfs.remove_file("dir_A/file_B")
    vfs.remove_dir("dir_A")


if __name__ == '__main__':
    main()