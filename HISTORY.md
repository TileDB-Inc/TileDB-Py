# TileDB-Py 0.6.0 Release Notes

## Improvements
* Target TileDB version 2.0
  - Added support for heterogeneous and string-typed dimensions [#304](https://github.com/TileDB-Inc/TileDB-Py/pull/304)
  - Added support for `tiledb_array_vacuum` for cleaning up consolidated fragments
* Added Windows wheels for Python 3.7 and 3.8 on PyPI

# TileDB-Py 0.5.9 Release Notes

* Bump release target to [TileDB 1.7.7](https://github.com/TileDB-Inc/TileDB/releases/tag/1.7.7)

# TileDB-Py 0.5.8 Release Notes

* Rebuild/release due to wheel build error on linux for 0.5.7.

# TileDB-Py 0.5.7 Release Notes

* Bump release target to [TileDB 1.7.6](https://github.com/TileDB-Inc/TileDB/releases/tag/1.7.6)

# TileDB-Py 0.5.6 Release Notes

* Bump release target to [TileDB 1.7.5](https://github.com/TileDB-Inc/TileDB/releases/tag/1.7.5)

# TileDB-Py 0.5.5 Release Notes

* Bump release target to [TileDB 1.7.4](https://github.com/TileDB-Inc/TileDB/releases/tag/1.7.4)

## Improvements
- Return coordinates by default for dense `multi_index` queries [#259](
https://github.com/TileDB-Inc/TileDB-Py/pull/259)

# TileDB-Py 0.5.4 Release Notes

* Bump release target to [TileDB 1.7.3](https://github.com/TileDB-Inc/TileDB/releases/tag/1.7.3)

## Improvements
- macOS wheels are now available on PyPI [#258](https://github.com/TileDB-Inc/TileDB-Py/pull/258)
- Delay default ctx initialization, allows per-process global config options to be controlled by user [#256](https://github.com/TileDB-Inc/TileDB-Py/pull/256)

# TileDB-Py 0.5.3 Release Notes

PyPI packages: https://pypi.org/project/tiledb/0.5.3/

## Improvements
- Reduce i/o overhead of `tiledb.open` and array constructors. [#239](https://github.com/TileDB-Inc/TileDB-Py/pull/239), [#240](https://github.com/TileDB-Inc/TileDB-Py/pull/240)
- Internal support for retrying incomplete queries in all array indexing modes. [#238](https://github.com/TileDB-Inc/TileDB-Py/pull/238), [#252](https://github.com/TileDB-Inc/TileDB-Py/pull/252)
- Eliminate reference cycles to improve Ctx cleanup. [#249](https://github.com/TileDB-Inc/TileDB-Py/pull/249)
- Support for retrieving compressor level from filter. [#234](https://github.com/TileDB-Inc/TileDB-Py/pull/234)

## Bug fixes
- Fix variable-length indexing error. [#236](https://github.com/TileDB-Inc/TileDB-Py/pull/236)
- Fix race condition initializing `tiledb.cloud` mixin from thread pool. [#246](https://github.com/TileDB-Inc/TileDB-Py/pull/246)

# TileDB-Py 0.5.2 Release Notes

## Bug fixes
- Fix bug in multi_index result buffer calculation [#232](https://github.com/TileDB-Inc/TileDB-Py/pull/232)

# TileDB-Py 0.5.1 Release Notes

## Bug fixes
- [Fix current buffer size calculation](https://github.com/TileDB-Inc/TileDB-Py/commit/3af75b5911b2195ceb66a41d582d9ffa9aa227b6)
- [Fix incorrect query_free in multi-range dense path](https://github.com/TileDB-Inc/TileDB-Py/commit/dbec665da3ebd0e0b5a341d22e47b25ede05cd7d)

## Other
- [Support '--tiledb=source' option for setup.py to ensure build from source](https://github.com/TileDB-Inc/TileDB-Py/commit/67e7c5c490caf97c5351352cb720116a1c5e1a0d)

# TileDB-Py 0.5.0 Release Notes

## New features
- add support for multi-range queries [#219](https://github.com/TileDB-Inc/TileDB-Py/pull/219)
- add support for TileDB array metadata [#213](https://github.com/TileDB-Inc/TileDB-Py/pull/213)
- add support for TILEDB_DATETIME_* attributes, domains, and slicing [#211](https://github.com/TileDB-Inc/TileDB-Py/pull/211)
- add support for retrieving list of fragments written by the most recent write to an array [#207](https://github.com/TileDB-Inc/TileDB-Py/pull/207)

## Bug fixes
- fix read error with multi-attribute sparse arrays [#214](https://github.com/TileDB-Inc/TileDB-Py/pull/214)

# TileDB-Py 0.4.4 Release Notes

* Bump release target to [TileDB 1.6.3](https://github.com/TileDB-Inc/TileDB/releases/tag/1.6.3)

## New features
- add `dim_type` keyword argument to `from_numpy` in order to override inferred Dimension dtype [#194](https://github.com/TileDB-Inc/TileDB-Py/pull/194)
- add `Array.domain_index`: slice over any range within the domain bounds, including negative slices [#202](https://github.com/TileDB-Inc/TileDB-Py/pull/202)

# TileDB-Py 0.4.3 Release Notes

* Bump release target to [TileDB 1.6.0](https://github.com/TileDB-Inc/TileDB/releases/tag/1.6.0)


## New features
- allow `tiledb.open` and `Array.create` to take an optional Ctx to override schema [#162](https://github.com/TileDB-Inc/TileDB-Py/pull/162)
- add `tiledb.array_exists` [#167](https://github.com/TileDB-Inc/TileDB-Py/pull/167)

## Bug fixes
- wrap query_submits into try / finally blocks correctly propagate KeyboardInterrupt errors while cleaning up resources [#155](https://github.com/TileDB-Inc/TileDB-Py/pull/155)
- fixed OOB access in exception handling path [#159](https://github.com/TileDB-Inc/TileDB-Py/pull/159)
- raise an error when trying to consolidate an open readonly array [#172](https://github.com/TileDB-Inc/TileDB-Py/pull/172)

# TileDB-Py 0.4.2 Release Notes

TileDB-Py 0.4.2 contains several improvements as well as bug-fixes associated with the TileDB 1.5.1 release.

## New features

- support for NumPy complex types ([#142](https://github.com/TileDB-Inc/TileDB-Py/pull/142))

## Bug fixes
- fixed query buffer memory leak ([#151](https://github.com/TileDB-Inc/TileDB-Py/pull/151))
- fixed segfault during consolidation ([TileDB #1213](https://github.com/TileDB-Inc/TileDB/pull/1213))
   - *note: to receive this fix, conda and source builds should be updated to TileDB 1.5.1. TileDB-Py 0.4.2 binaries on PyPI bundle the updated TileDB 1.5.1 library.*
- fixed indexing with array dtype different from the platform default ([#146](https://github.com/TileDB-Inc/TileDB-Py/pull/146))
- fixed `VFS.is_bucket` when VFS is initialized with a Ctx object ([#148](https://github.com/TileDB-Inc/TileDB-Py/pull/148))
- fixed `schema_like` to correctly forward a Ctx keyword arg ([#148](https://github.com/TileDB-Inc/TileDB-Py/pull/148))

# TileDB-Py 0.4.1 Release Notes

## New Features:

* several high-level API additions (tiledb.open, .save, .empty_like, .schema_like), and serialization improvements including pickling support for DenseArray objects (#129)
* manylinux1 wheels for Python 2.7, 3.5, 3.6, and 3.7 are available on PyPI: https://pypi.org/project/tiledb

# TileDB-Py 0.4.0 Release Notes

This release builds TileDB-Py against TileDB 1.5

## New Features:

* support for variable-length arrays (#120)

## Breaking changes:

* the Ctx argument is now a keyword argument, simplifying API use in the common case (#122)

  for example: `tiledb.DenseArray(ctx, uri, ...)` becomes: tiledb.DenseArray(uri, ...)
  or optionally `tiledb.DenseArray(uri, ..., ctx=ctx)`
