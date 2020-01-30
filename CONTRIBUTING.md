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

- [Please follow these instructions to build from source](https://docs.tiledb.com/developer/installation/building-from-source/python)
- Make changes locally, then rebuild with `python setup.py develop`
- Make sure to run `pytest` to verify changes against tests (add new tests where applicable).
- Please submit [pull requests](https://help.github.com/en/desktop/contributing-to-projects/creating-a-pull-request) against the default [`dev` branch of TileDB-Py](https://github.com/TileDB-Inc/TileDB-Py/tree/dev)