TileDB for Python
#################
`tiledb` is a pythonic interface to the `TileDB array storage manager <http://tiledb.io>`_.


Building TileDB-Py
==================

Dependencies
------------
* Numpy

Installing
----------

Testing TileDB-Py from within the source folder
-----------------------------------------------

TileDB-Py can be tested without having the package installed.
To compile the sources inplace from the source directory:

    $ python setup.py build_ext --inplace

Tests can now be run using Python's unittest framework

    $ python -m unittest

Youo can also install a `symlink named site-packages/tiledb.egg-link` to the development folder of TileDB-Py with:

    $ pip install --editable .

This enables local changes to the current development repo to be reflected globally
