# Contributing to TileDB-Py

Thanks for your interest in TileDB-Py. The notes below give some pointers for filing issues and bug reports, or contributing to the code.

## Contribution Checklist
- Reporting a bug? Please include the following information
  - operating system and version (windows, linux, macos, etc.)
  - the output of `tiledb.version()` and `tiledb.libtiledb.version()`
  - if possible, a minimal working example demonstrating the bug or issue (along with any data to re-create, when feasible)
- Please paste code blocks with triple backquotes (```) so that github will format it nicely. See [GitHub's guide on Markdown](https://guides.github.com/features/mastering-markdown) for more formatting tricks.

## Contributing Code
*By contributing code to TileDB-Py, you are agreeing to release it under the [MIT License](https://github.com/TileDB-Inc/TileDB/tree/dev/LICENSE).*

### Contribution Workflow

- Quick steps to build locally:
  - install prerequisites via pip or conda: `pybind11` `cython` `numpy` `pandas` `pyarrow`
  - recommended: install TileDB embedded (libtiledb)
    
    NOTE: if libtiledb path is not specified with `--tiledb`, it will be built automatically by `setup.py`. However, this build
          is internal to the source tree and somewhat difficult to modify. When working on both projects simultaneously, it is
          strongly suggested to build libtiledb separately. Changes to libtiledb must be `make install-tiledb` to `dist` in
          order to be used with `--tiledb`.
            
    - from latest release build: https://github.com/TileDB-Inc/TileDB/releases
      - `tar xf tiledb-<platform>-<hash>.tar.gz -C /path/to/extract`
      - use `--tiledb=/path/to/extract` (note: this path should _contain_ the `lib` directory)
    - from [conda-forge](): `mamba install tiledb`
      - `--tiledb=$CONDA_PREFIX`
    - from source: https://docs.tiledb.com/main/how-to/installation/building-from-source/c-cpp
      - use `--tiledb=/path/to/tiledb/dist` option when running ``setup.py`` in the step below
      - if building libtiledb from source,  to enable serialization pass ``--enable-serialization`` 
        to the ``bootstrap`` script before compiling
	- serialization is optional. if libtiledb is not build with serialization, then it will not be
	  enabled in TileDB-Py
            
  - build TileDB-Py
  ```
  git clone https://github.com/TileDB-Inc/TileDB-Py
  cd TileDB-Py
  python setup.py develop --tiledb=</path/to/tiledb/dist>
  ```

- Make changes locally, then rebuild with `python setup.py develop [--tiledb=<>]`
- Make sure to run `pytest` to verify changes against tests (add new tests where applicable).
  - Execute the tests as `pytest tiledb` from the top-level directory or `pytest` in the `tiledb/` directory.
- Please submit [pull requests](https://help.github.com/en/desktop/contributing-to-projects/creating-a-pull-request) against the default [`dev` branch of TileDB-Py](https://github.com/TileDB-Inc/TileDB-Py/tree/dev)
