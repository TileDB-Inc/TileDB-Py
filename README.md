<a href="https://tiledb.com"><img src="https://github.com/TileDB-Inc/TileDB/raw/dev/doc/source/_static/tiledb-logo_color_no_margin_@4x.png" alt="TileDB logo" width="400"></a>


[![Build Status](https://dev.azure.com/TileDB-Inc/CI/_apis/build/status/TileDB-Inc.TileDB-Py?branchName=dev)](https://dev.azure.com/TileDB-Inc/CI/_build/latest?definitionId=1&branchName=dev)
![](https://raw.githubusercontent.com/TileDB-Inc/TileDB/dev/doc/anaconda.svg?sanitize=true)[![Anaconda download count badge](https://anaconda.org/conda-forge/TileDB-Py/badges/downloads.svg)](https://anaconda.org/conda-forge/TileDB-Py)


# TileDB-Py

*TileDB-Py* is a [Python](https://python.org) interface to the [TileDB Storage Engine](https://github.com/TileDB-Inc/TileDB).

# Quick Links

* [Installation](https://docs.tiledb.com/developer/installation/quick-install)
* [Build Instructions](https://docs.tiledb.com/main/how-to/installation/building-from-source/python)
* [TileDB Documentation](https://docs.tiledb.com/main/)
* [Python API reference](https://tiledb-inc-tiledb-py.readthedocs-hosted.com/en/stable)

# Quick Installation

TileDB-Py is available from either [PyPI](https://pypi.org/project/tiledb/) with ``pip``:

```
pip install tiledb
```

or from [conda-forge](https://anaconda.org/conda-forge/tiledb-py) with
[conda](https://conda.io/docs/) or [mamba](https://github.com/mamba-org/mamba#installation):

```
conda install -c conda-forge tiledb-py
```

Dataframes functionality (`tiledb.from_pandas`, `Array.df[]`) requires [Pandas](https://pandas.pydata.org/) 1.0 or higher, and [PyArrow](https://arrow.apache.org/docs/python/) 1.0 or higher.

# Contributing

We welcome contributions, please see [`CONTRIBUTING.md`](CONTRIBUTING.md) for suggestions and
development-build instructions. For larger features, please open an issue to discuss goals and
approach in order to ensure a smooth PR integration and review process.
