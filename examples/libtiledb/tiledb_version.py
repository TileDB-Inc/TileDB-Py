#!/usr/bin/python

"""
Prints the version of libtiledb (TileDB's shared library)
"""

import tiledb

def main():
    major, minor, patch = tiledb.libtiledb.version()
    print("TileDB v{major}.{minor}.{patch}"
          .format(major=major, minor=minor, patch=patch))

if __name__ == '__main__':
    main()
