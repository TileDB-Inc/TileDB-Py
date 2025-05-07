# Building TileDB-Py from Source

## Build dependencies

* NumPy
* Cython
* pybind11
* scikit-build-core
* C++20 compiler
* CMake

## Runtime Dependencies

* NumPy

### macOS and Linux

Execute the following commands:

```bash
$ git clone https://github.com/TileDB-Inc/TileDB-Py.git
$ cd TileDB-Py
$ pip install .
$ cd .. # exit the source directory to avoid import errors
```

If you wish to modify the install process, you can use these environment variables:

* `TILEDB_PATH`: Path to TileDB core library. If this variable is set and the library is found in the specified folder it is not copied inside of the wheel.
* `TILEDB_VERSION`: Version of the TileDB core library that you wish to download. This version must be present in the Github releases.
* `TILEDB_HASH`: SHA256 sum of the desired TileDB core library release. Only used when `TILEDB_VERSION` is set.

```bash
$ TILEDB_PATH=/home/tiledb/dist pip install .
# Or pass it as an argument
$ pip install . -C skbuild.cmake.define.TILEDB_PATH=/home/tiledb/dist
```

To build against `libtiledb`  installed with conda, run:

```bash
# After activating the desired conda environment
$ conda install tiledb
$ TILEDB_PATH=${PREFIX} python -m pip install --no-build-isolation --no-deps --ignore-installed -v .
```

To test your local installation, install optional dependencies, and then use `pytest`:

```
$ pip install .[test]
$ python -m pytest -v # in the TileDB-Py source directory
```

If TileDB is installed in a non-standard location, you also need to make the dynamic linker aware of `libtiledb`'s location. Otherwise when importing the `tiledb` module you will get an error that the built extension module cannot find `libtiledb`'s symbols:

```
$ env LD_LIBRARY_PATH="/home/tiledb/dist/lib:$LD_LIBRARY_PATH" python -m pytest -v
```

For macOS the linker environment variable is `DYLD_LIBRARY_PATH`**.**

### Windows

If you are building the extension on Windows, first install a Python distribution such as [Miniconda](https://conda.io/miniconda.html). You can then either build TileDB from source, or download the pre-built binaries.

Once you've installed Miniconda and TileDB, execute:

```bash
REM with a conda install of libtiledb:
> pip install .

REM with a TileDB source build:
> set TILEDB_PATH=C:/path/to/TileDB/dist/bin
> pip install .

REM to run tests:
> pip install .[test]
> python -m pytest -v
```

Note that if you built TileDB locally from source, then replace `set TILEDB_PATH=C:/path/to/TileDB` with `TILEDB_PATH=C:/path/to/TileDB/dist`.
