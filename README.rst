.. image:: https://github.com/TileDB-Inc/TileDB/raw/dev/doc/source/_static/tileDB_uppercase_600_112.png
    :target: https://tiledb.io
    :alt: TileDB logo
    :width: 400

|

.. image:: https://travis-ci.org/TileDB-Inc/TileDB-Py.svg?branch=dev
    :target: https://travis-ci.org/TileDB-Inc/TileDB-Py
    :alt: Travis status

.. image:: https://readthedocs.com/projects/tiledb-inc-tiledb-py/badge/?version=latest
    :target: https://tiledb-inc-tiledb-py.readthedocs-hosted.com/en/latest/?badge=latest
    :alt: Documentation Status

|

**Array data management made fast and easy.**

`TileDB <https://tiledb.io>`_ is an efficient multi-dimensional array management system which introduces a novel on-disk format that can effectively store dense and sparse array data with support for fast updates and reads. It also features excellent compression and an efficient parallel I/O system with high scalability.

**TileDB-Py** is the official Python interface to TileDB.

Quickstart
----------

First, install TileDB-Py with ``pip``::

    $ pip install tiledb

This may take a while, as the pip package will automatically download and build the native TileDB library in addition to the Python bindings.

Next, save the `quickstart program <https://github.com/TileDB-Inc/TileDB-Py/blob/dev/examples/quickstart_dense.py>`_ into a file and run it::

    $ wget https://github.com/TileDB-Inc/TileDB-Py/blob/dev/examples/quickstart_dense.py
    $ python quickstart_dense.py
    [[2 3 4]
     [6 7 8]]

The dense quickstart program simply creates a dense array on disk, writes some simple data to it, and reads a slice of the data back, printing the slice to the console.

Documentation
-------------

The full TileDB documentation can be found at `<https://docs.tiledb.io>`_ and includes many tutorials and examples to get you started.

The latest Python API reference can be found at `<https://docs.tiledb.io/projects/tiledb-py/en/latest/python-api.html>`_.

Installation
------------

Pip
~~~

A PyPI package is available which can be installed with Pip. This package will download and install the native TileDB library inside the site package if TileDB is not already installed on your system.

::

    $ pip install tiledb

Note: if the Numpy and Cython dependencies are not installed, pip will try to build them from source.  This can take a **long** time and make the install appear to "hang."  Pass the ``-v`` flag to pip to monitor the build process.

If you wish to use a custom version of the TileDB library and the install location is not in the compiler search path, create a requirements.txt file that specifies the tiledb install path manually.

::

    $ cat > tiledb_requirements.txt <<EOF
      tiledb==<version> --install-option="--tiledb=<path/to/tiledb/install>"
      EOF
    $ pip install -r tiledb_requirements.txt

Do not forget to put the built ``.so / .dylib / .dll`` on the dynamic linker path, otherwise TileDB-Py will fail to load the shared library upon import.


Conda Package
~~~~~~~~~~~~~

A pre-built Conda package is available that will install TileDB as well.

::

    $ conda install -c conda-forge tiledb-py

Note: Currently the pre-built TileDB conda package does not include the HDFS and S3 storage backends.

Installing From Source
~~~~~~~~~~~~~~~~~~~~~~

TileDB-Py Build Dependencies
''''''''''''''''''''''''''''

* Numpy
* Cython
* C++11 compiler
* CMake

TileDB-Py Runtime Dependencies
''''''''''''''''''''''''''''''

* Numpy

Linux / OSX
'''''''''''

Simply execute the following commands::

   $ git clone https://github.com/TileDB-Inc/TileDB-Py.git
   $ cd TileDB-Py
   $ pip install -r requirements_dev.txt
   $ python setup.py build_ext --inplace
   $ python setup.py install

If you wish to use a custom version of the TileDB library and it is installed in a non-standard location, pass the path to ``setup.py`` with the ``--tiledb=`` flag.
If you want to pass extra compiler/linker flags during the C++ extension compilation step use ``--cxxflags=`` or ``--lflags=``.

::

  $ python setup.py build_ext --inplace --tiledb=/home/tiledb/dist 

If TileDB is installed in a non-standard location, you also need to make the dynamic linker aware of ``libtiledb``'s location.
Otherwise when importing the ``tiledb`` module you will get an error that the built extension module cannot find
``libtiledb``'s symbols::

  $ env LD_LIBRARY_PATH="/home/tiledb/dist/lib:$LD_LIBRARY_PATH" python -m unittest -v

For macOS the linker environment variable is ``DYLD_LIBRARY_PATH``

Installing on Windows
'''''''''''''''''''''

If you are building the extension on Windows, first install a Python distribution such as `Miniconda <https://conda.io/miniconda.html>`_. You can then either build TileDB from source, or download the pre-built binaries.

Once you've installed Miniconda and TileDB, open the Miniconda command prompt and execute:

::

   > cd TileDB-Py
   > conda install conda-build
   > conda install virtualenv
   > virtualenv venv
   > venv\Scripts\activate
   > pip install -r requirements_dev.txt
   > python setup.py build_ext --inplace --tiledb=C:\path\to\TileDB\
   > set PATH=%PATH%;C:\path\to\TileDB\bin
   > python -m unittest -v

Note that if you built TileDB from source, then replace ``C:\path\to\TileDB`` with ``C:\path\to\TileDB\dist``.

Developing and testing TileDB-Py
--------------------------------

TileDB-Py includes a handy Conda environment definition file for setting up a test environment::

    $ conda env create -f environment.yml

This will create a ``tiledbpy`` conda environment with all the development library dependencies.

The easiest way to test / develop TileDB-Py across Python versions (2.7, 3.5, and 3.6),
is using `tox <https://tox.readthedocs.io/en/latest/index.html>`_.
TileDB includes a tox.ini file, simply run ``tox`` in the toplevel source directory to run the test suite against multiple installed Python versions::

    $ tox

You can specify a particular Python version using the ``-e`` flag::

    $ tox -e py27

If TileDB is not installed in a global system location, you must specify the install path to tox::

    $ env TILEDB_PATH=/path/to/tiledb LD_LIBRARY_PATH=/path/to/tiledb/libdir:${LD_LIBRARY_PATH} tox

You can also run the unittests from the source folder without having the package installed. First build the package in place from the source directory::

    $ python setup.py build_ext --inplace

Tests can now be run using Python's unittest framework::

    $ python -m unittest -v

Doctests can be run using the doctest modele::

    $ python -m doctest -o NORMALIZE_WHITESPACE -f tiledb/libtiledb.pyx

You can also install a symlink named ``site-packages/tiledb.egg-link`` to the development folder of TileDB-Py with::

    $ pip install --editable .

This enables local changes to the current development repo to be reflected globally.
