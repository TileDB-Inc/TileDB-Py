#!/usr/bin/python

"""
It creates a hierarchical directory structure with three groups:
     my_group
        |_ dense_arrays
        |_ sparse_arrays

Make sure that no folder with name `my_group` exists in the working
directory before running this example.
"""

import tiledb


def main():
    ctx = tiledb.Ctx()


if __name__ == '__main__':
    main()
