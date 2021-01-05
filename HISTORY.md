# TileDB-Py 0.7.5 Release Notes

## Packaging Notes
* TileDB-Py 0.7.x will be the last version of TileDB-Py supporting Python 2.

## TileDB Embedded updates:
* TileDB-Py 0.7.5 includes [TileDB Embedded 2.1.6](https://github.com/TileDB-Inc/TileDB/releases/tag/2.1.6)

## Improvements
* FragmentInfo API by default returns information from all fragments and dimensions [#444](https://github.com/TileDB-Inc/TileDB-Py/pull/444)
* Add integer multi-indexing for NumPy datetime64 dimensions [#447](https://github.com/TileDB-Inc/TileDB-Py/pull/447) 
* Add `from_csv/pandas` support for `timestamp` keyword argument to specify write timestamp [#450](https://github.com/TileDB-Inc/TileDB-Py/pull/450)
* Add verbosity option to `stats_dump()` [#452](https://github.com/TileDB-Inc/TileDB-Py/pull/452)
* Add `unique_dim_values()` to return unique dimension values for a given `SparseArray` [#454](https://github.com/TileDB-Inc/TileDB-Py/pull/454)
* Add support to `query()` for returning subsets of specified dimensions [#458](https://github.com/TileDB-Inc/TileDB-Py/pull/458)
* Optimize string array writes [#459](https://github.com/TileDB-Inc/TileDB-Py/pull/459)

## Bug fixes
* Fix `Dim.shape` for dense array with datetime dimension [#448](https://github.com/TileDB-Inc/TileDB-Py/pull/448)

# TileDB-Py 0.7.4 Release Notes

## Improvements
* Support selecting subset of dimensions in Array.query via new keyword argument `dims: List[String]`. The `coords=True` kwarg is still supported for compatibility, and continues to return all dimensions [#433](https://github.com/TileDB-Inc/TileDB-Py/pull/433)
* Support Dim(filters=FilterList) keyword argument to set filters on a per-Dim basis [#434](https://github.com/TileDB-Inc/TileDB-Py/pull/434)
* Support tiledb.from_csv setting attribute and dimension filters by dictionary of {name: filter} [#434](https://github.com/TileDB-Inc/TileDB-Py/pull/434)
* Add ArraySchema.check wrapping `tiledb_array_schema_check` [#435](https://github.com/TileDB-Inc/TileDB-Py/pull/435)
* Add support for attribute fill values `tiledb.Attr(fill=...)` and `Attr.fill` getter [#437](https://github.com/TileDB-Inc/TileDB-Py/pull/437)

## API Changes
* tiledb.from_csv keyword arg `attrs_filters` renamed to `attr_filters` [#434](https://github.com/TileDB-Inc/TileDB-Py/pull/434)

## Bug fixes
* Fix bug in `multi_index` slicing of dense arrays [#438](https://github.com/TileDB-Inc/TileDB-Py/pull/438)

# TileDB-Py 0.7.3 Release Notes

## Improvements
* The default result layout for indexing/querying sparse arrays is now TILEDB_UNORDERED [#428](https://github.com/TileDB-Inc/TileDB-Py/pull/428), [#431](https://github.com/TileDB-Inc/TileDB-Py/pull/431)
* Added documentation for all TileDB-Py configuration parameters [#430](https://github.com/TileDB-Inc/TileDB-Py/pull/430)
* Fixed documentation rendering for `Array.query` [#430](https://github.com/TileDB-Inc/TileDB-Py/pull/430)

## Bug fixes
* Fix sparse dimension type selection when array type is not specified to from_pandas [#429](https://github.com/TileDB-Inc/TileDB-Py/pull/429)
* Don't pass allows_duplicates=True to dense array constructor (tiledb.from_csv) [#428](https://github.com/TileDB-Inc/TileDB-Py/pull/428)

# TileDB-Py 0.7.2 Release Notes

## TileDB Embedded updates:
* TileDB-Py 0.7.2 includes [TileDB Embedded 2.1.3](https://github.com/TileDB-Inc/TileDB/releases/tag/2.1.3)
  Including a fix for issue [#409](https://github.com/TileDB-Inc/TileDB-Py/issues/409).

## Changes
* The default array type for `from_pandas` and `from_csv` is now dense, if unspecified, except when passing a dataframe with string indexes to `from_pandas` [#424](https://github.com/TileDB-Inc/TileDB-Py/pull/408)

## Improvements
* Automatically determine column to dimension mapping for `tiledb.from_csv` append mode [#408](https://github.com/TileDB-Inc/TileDB-Py/pull/408)

## Bug fixes
* Fixed `tiledb.from_csv/dataframe` error when ingesting single-row/index datasets [#422]()
* Fixed intermittent `csv_sparse_col_to_dims` failure due to duplicate result ordering [#423](https://github.com/TileDB-Inc/TileDB-Py/pull/423)

# TileDB-Py 0.7.1 Release Notes

## Improvements
* Added support for `df[]` indexing via `tiledb.Array.query` [#411](https://github.com/TileDB-Inc/TileDB-Py/pull/411)
* Modified `stats_dump` to return internal stats as string, allowing for output in Jupyter notebooks [#403](https://github.com/TileDB-Inc/TileDB-Py/pull/403)
* Added `__repr__` to `Array` and `Ctx` [#413](https://github.com/TileDB-Inc/TileDB-Py/pull/413)
* `tiledb.open` now supports `timestamp` keyword argument [#419](https://github.com/TileDB-Inc/TileDB-Py/pull/419)
* `tiledb.Domain` now supports passing a list of `Dim`s without unpacking [#419](https://github.com/TileDB-Inc/TileDB-Py/pull/419)

## Bug fixes
* Fixed PyPI wheels load error on newer macOS due to overlinkage against system libraries in build process (curl -> libintl) [#418](https://github.com/TileDB-Inc/TileDB-Py/pull/418)
* Fixed PyPI wheels load error on Windows due to building against TBB [#419](https://github.com/TileDB-Inc/TileDB-Py/pull/419)
* Fixed indexing of attribute named 'coords' [#414](https://github.com/TileDB-Inc/TileDB-Py/pull/414)
* `open_dataframe` now uses the underlying Array's `nonempty_domain` to avoid errors opening unlimited domain arrays [#409](https://github.com/TileDB-Inc/TileDB-Py/pull/409)

# TileDB-Py 0.7.0 Release Notes

## TileDB Embedded updates:
* TileDB-Py 0.7.0 includes [TileDB Embedded 2.1.2](https://github.com/TileDB-Inc/TileDB/releases/tag/2.1.2)
  featuring a number of significant improvements, with major highlights including:
  - no longer uses Intel TBB for parallelization by default. Along with many benefits to TileDB Embedded, this
    significantly reduces complications and bugs with python multiprocessing fork mode.
  - Support coalescing subarray ranges to give major performance boosts.

## Packaging Notes
* TileDB-Py 0.7 packages on PyPI support macOS 10.13+ and manylinux10-compatible Linux distributions only.
  For now, wheels could be produced supporting older systems but without Google Cloud Support; if needed,
  please contact us to discuss.

## Improvements
* Added ".df[]" indexer tiledb.Array: directly returns a Pandas dataframe from a query (uses `multi_index` indexing behavior) [#390](https://github.com/TileDB-Inc/TileDB-Py/pull/389)
* Added parallel CSV ingestion example using Python multiprocessing with `tiledb.from_csv` [#397](https://github.com/TileDB-Inc/TileDB-Py/pull/397)
* Added wrapping and support for TileDB checksumming filters: `ChecksumMD5Filter` and `ChecksumSHA256Filter` [#389](https://github.com/TileDB-Inc/TileDB-Py/pull/389)
* Removed TBB install from default setup.py, corresponding to TileDB Embedded changes [#389](https://github.com/TileDB-Inc/TileDB-Py/pull/389)
* Add support for 'capacity' kwarg to `from_csv`/`from_pandas` [#391](https://github.com/TileDB-Inc/TileDB-Py/pull/391)
* Add support for 'tile' kwarg to `from_csv`/`from_pandas` to customize Dim tile extent [#391](https://github.com/TileDB-Inc/TileDB-Py/pull/391)
* Added '--release-symbols' option for building in release optimization with debug symbols [#402](https://github.com/TileDB-Inc/TileDB-Py/pull/402)
* Changed `allows_duplicates` default to `True` for `from_csv/from_pandas` [#394](https://github.com/TileDB-Inc/TileDB-Py/pull/394)

## Bug fixes
* Fixed bug indexing anonymous attributes of sparse arrays using `A[]` (did not affect dense or multi_index) [#404](https://github.com/TileDB-Inc/TileDB-Py/pull/404)
* Fixed rendering of column name in mixed dtype exception [#382](https://github.com/TileDB-Inc/TileDB-Py/pull/382)
* Fixed forwarding of 'ctx' kwarg to from_csv/from_pandas [#383](https://github.com/TileDB-Inc/TileDB-Py/pull/383)
* Fixed type of return values for empty results when indexing a sparse array [#384](https://github.com/TileDB-Inc/TileDB-Py/pull/384)

## Misc Updates
* Added round-trip tests for all filter `repr` objects [#389](https://github.com/TileDB-Inc/TileDB-Py/pull/389)

# TileDB-Py 0.6.6 Release Notes

**Note that we will be removing wheel support for macOS 10.9-10.12 in TileDB-Py 0.7 (planned for release in August 2020).** This change is due to upstream (AWS SDK) minimum version requirements. The minimum supported version for macOS wheels on PyPI will be macOS 10.13.

**Note that we will be removing support for [manylinux1](https://github.com/pypa/manylinux/tree/manylinux1) wheels in TileDB-Py 0.7 (planned for release in August 2020).** manylinux1 is based on CentOS5, which has been unsupported for several years. We now provide wheels built with [manylinux2010](https://www.python.org/dev/peps/pep-0571/), which is based on CentOS6 / glibc 2.12.

## Improvements
* Bump release target to [TileDB 2.0.7](https://github.com/TileDB-Inc/TileDB/releases/tag/2.0.7)

# TileDB-Py 0.6.5 Release Notes

We have added manylinux2010 wheels, corresponding to CentOS6 / glibc 2.12.

We are deprecating support for manylinux1 (CentOS5 / glibc 2.0.7), which is not supported by
the Google Cloud Storage SDK. We are planning to remove manylinux1 wheel support in the
TileDB-Py 0.7 release.


## Improvements
* Enabled Google Cloud Storage support in macOS and linux (manylinux2010) wheels on PyPI ([#364](https://github.com/TileDB-Inc/TileDB-Py/pull/364))

# TileDB-Py 0.6.4 Release Notes

## API notes
* Deprecated `initialize_ctx` in favor of `default_ctx(config: tiledb.Config)` [#351](https://github.com/TileDB-Inc/TileDB-Py/pull/351)

## Improvements
* Bump release target to [TileDB 2.0.6](https://github.com/TileDB-Inc/TileDB/releases/tag/2.0.6)
* Improved error reporting for input data type mismatches [#359](https://github.com/TileDB-Inc/TileDB-Py/pull/359)
* Added `tiledb.VFS.dir_size` [#343](https://github.com/TileDB-Inc/TileDB-Py/pull/343)
* Added read and buffer conversion statistics for python to `tiledb.stats_dump` [#354](https://github.com/TileDB-Inc/TileDB-Py/pull/354)
* Implemented string deduplication to reduce conversion time for string arrays [#357](https://github.com/TileDB-Inc/TileDB-Py/pull/357)

## Bug fixes
* Fixed argument order for `Array.consolidate` with a Config override parameter [#344](https://github.com/TileDB-Inc/TileDB-Py/pull/344)

# TileDB-Py 0.6.3 Release Notes

## Improvements
* Bump release target to [TileDB 2.0.5](https://github.com/TileDB-Inc/TileDB/releases/tag/2.0.5)

## Bug fixes
* Fix unnecessary implicit ordering requirement for multi-attribute assignment. [#328](https://github.com/TileDB-Inc/TileDB-Py/pull/328)

# TileDB-Py 0.6.2 Release Notes

## Bug fixes
* Fix `nonempty_domain` with heterogeneous non-string dimensions ([#320](https://github.com/TileDB-Inc/TileDB-Py/pull/320))

## Improvements
* Add doctest for `tiledb.vacuum` ([#319](https://github.com/TileDB-Inc/TileDB-Py/pull/320))

# TileDB-Py 0.6.1 Release Notes

## Bug fixes
* Fix assignment order for `nonempty_domain` with string dimensions ([#308](https://github.com/TileDB-Inc/TileDB-Py/pull/308)) (test in [#311](https://github.com/TileDB-Inc/TileDB-Py/commit/35e5ff64ccfe7bf8f30a5900bfbe67c46cd1f97d))
* Fix bug in string attribute handling for var-length attributes ([#307](https://github.com/TileDB-Inc/TileDB-Py/issues/307))
* Fix regression reading anonymous attributes from TileDB 1.7 arrays ([#311](https://github.com/TileDB-Inc/TileDB-Py/pull/311))
* Fix incorrect `multi_index` error when string attribute results are empty ([#311](https://github.com/TileDB-Inc/TileDB-Py/pull/311))

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
