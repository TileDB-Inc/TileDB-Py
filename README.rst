TileDB for Python
#################
.. image:: https://travis-ci.org/TileDB-Inc/TileDB-Py.svg?branch=dev
    :target: https://travis-ci.org/TileDB-Inc/TileDB-Py


``TileDB-Py`` is a Python interface to the `TileDB array storage manager <https://tiledb.io>`_.


Runtime Dependencies
------------
* Numpy

Build Dependencies
------------------
* Numpy
* Cython
* C++11 compiler

Install
=======

You will need to build / install an up-to-date version of TileDB. 
See https://docs.tiledb.io/docs/installation for instructions.

Pip
'''

A PyPI package is available which can be installed with Pip:

:: 

    $ pip install tiledb
    
**Note** if the Numpy and Cython dependencies are not installed, pip will try to build them from source.  This can take a **long** time and make the install appear to "hang."  Pass the ``-v`` flag to pip to monitor the build process.

If the install location of TileDB is not in compiler search path, use the ``--install-option`` flag to manually pass the tiledb install location to ``setup.py``.

::
    
    $ pip install tiledb --install-option="--tiledb=C:\path\to\TileDB\bin"


Installing TileDB-Py from source
''''''''''''''''''''''''''''''''


::

   $ git clone https://github.com/TileDB-Inc/TileDB-Py.git
   $ cd TileDB-Py
   $ pip install -r requirements_dev.txt
   $ python setup.py build_ext --inplace
   $ python setup.py install

or simply

::

   $ pip install tiledb

If TileDB is installed in a non-standard location, pass the path to `setup.py` with the ``--tiledb=`` flag.
If you want to pass extra compiler/linker flags during the c++ extension compilation step use ``--cxxflags=`` or ``--lflags=``.

::

  $ python setup.py build_ext --inplace --tiledb=/home/tiledb/dist 

If TileDB is installed in a non-standard location, you need to make the dynamic linker aware of ``libtiledb``'s location.
Otherwise when importing the `tiledb` module you will get an error that the built extension module cannot find
``libtiledb``'s symbols.

::

  $env LD_LIBARY_PATH="/home/tiledb/dist:$LD_LIBRARY_PATH" python -m unittest -v


For macOS the linker env variable is ``DYLD_LIBARAY_PATH``

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

Testing TileDB-Py from within the source folder
-----------------------------------------------

TileDB-Py can be tested without having the package installed.
To compile the sources inplace from the source directory:

::

    $ python setup.py build_ext --inplace

Tests can now be run using Python's unittest framework

::

    $ python -m unittest -v

Youo can also install a `symlink named site-packages/tiledb.egg-link` to the development folder of TileDB-Py with:

::

    $ pip install --editable .

This enables local changes to the current development repo to be reflected globally
