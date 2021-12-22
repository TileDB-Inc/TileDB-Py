# TileDB-Py 0.11.4 Release Notes

## TileDB Embedded updates:
* TileDB-Py 0.11.4 includes TileDB Embedded [TileDB 2.5.3](https://github.com/TileDB-Inc/TileDB/releases/tag/2.5.3)

## API Changes
* Add rich display for TileDB objects in Jupyter notebooks [#824](https://github.com/TileDB-Inc/TileDB-Py/pull/824)
* Support `TILEDB_STRING_ASCII` for array metadata [#828](https://github.com/TileDB-Inc/TileDB-Py/pull/828)

# TileDB-Py 0.11.3 Release Notes

## Impovements
* Support for Python 3.10 [#808](https://github.com/TileDB-Inc/TileDB-Py/pull/808)

## API Changes
* Addition of `tiledb.version()` to return version as a tuple [#801](https://github.com/TileDB-Inc/TileDB-Py/pull/801)
* `Query.get_stats` and `Ctx.get_stats` changed function signature; automatically `print_out` stats and add option to output as `json` [#809](https://github.com/TileDB-Inc/TileDB-Py/pull/809)

## Bug fixes
* `tiledb.delete_fragments` removes unused schemas [#813](https://github.com/TileDB-Inc/TileDB-Py/pull/813)

# TileDB-Py 0.11.2 Release Notes

## TileDB Embedded updates:
* TileDB-Py 0.11.2 includes TileDB Embedded [TileDB 2.5.2](https://github.com/TileDB-Inc/TileDB/releases/tag/2.5.2)

## Bug fixes
* Support dict parameter for 'config' argument to VFS constructor [#805](https://github.com/TileDB-Inc/TileDB-Py/pull/805)

# TileDB-Py 0.11.1 Release Notes
 
## TileDB Embedded updates:
* TileDB-Py 0.11.1 includes TileDB Embedded [TileDB 2.5.1](https://github.com/TileDB-Inc/TileDB/releases/tag/2.5.1)

## Bug Fixes
* Correct libtiledb version checking for Fragment Info API getters' MBRs and array schema name [#784](https://github.com/TileDB-Inc/TileDB-Py/pull/784)
* 
# TileDB-Py 0.11.0 Release Notes

## TileDB Embedded updates:
* TileDB-Py 0.11.0 includes TileDB Embedded [TileDB 2.5.0](https://github.com/TileDB-Inc/TileDB/releases/tag/2.5.0)

## API Changes
* Addition of MBRs to `FragmentInfo` API [#760](https://github.com/TileDB-Inc/TileDB-Py/pull/760)
* Addition of `array_schema_name` to `FragmentInfo` API [#777](https://github.com/TileDB-Inc/TileDB-Py/pull/777)
* Addition of `tiledb.create_array_from_fragments` to copy fragments within a given timestamp range to a new array [#777](https://github.com/TileDB-Inc/TileDB-Py/pull/777)

# TileDB-Py 0.10.5 Release Notes

## API Changes
* Addition of `tiledb.delete_fragments` to remove fragments within a given timestamp range [#774](https://github.com/TileDB-Inc/TileDB-Py/pull/774)

# TileDB-Py 0.10.4 Release Notes

## TileDB Embedded updates:
* TileDB-Py 0.10.4 includes TileDB Embedded [TileDB 2.4.3](https://github.com/TileDB-Inc/TileDB/releases/tag/2.4.3)

## Bug fixes
* Error out when applying `QueryCondition` to dense arrays; this feature will be implemented in TileDB Embedded 2.5 [#753](https://github.com/TileDB-Inc/TileDB-Py/pull/753)
* Ensure that indexer, multi-indexer, and .df return the same results when applying `QueryCondition` [#753](https://github.com/TileDB-Inc/TileDB-Py/pull/753)
* Fix error when using .df with PyArrow 6 due to incorrect metadata field in exported schema [#764](https://github.com/TileDB-Inc/TileDB-Py/pull/764)
* Fix  [#755](https://github.com/TileDB-Inc/TileDB-Py/issues/755): `from_pandas` to correctly round-trip unnamed Index [#761](https://github.com/TileDB-Inc/TileDB-Py/pull/761)
* Fix .df indexer bug with empty result set [#744](https://github.com/TileDB-Inc/TileDB-Py/pull/744)

## API Changes
* Close the `PyFragmentInfo` object in the `FragmentInfoList` constructor to reflect changes in the `FragmentInfo` API in TileDB Embedded 2.5 [#752](https://github.com/TileDB-Inc/TileDB-Py/pull/752)
* Make `ctx` argument optional for `ArraySchemaEvolution` [#743](https://github.com/TileDB-Inc/TileDB-Py/pull/743)
* Remove `coords_filters` from `ArraySchema` for dense arrays [#762](https://github.com/TileDB-Inc/TileDB-Py/pull/762)

# TileDB-Py 0.10.3 Release Notes

## TileDB Embedded updates:
* TileDB-Py 0.10.3 includes TileDB Embedded [TileDB 2.4.2](https://github.com/TileDB-Inc/TileDB/releases/tag/2.4.2)
  - Note that 2.4.1 was skipped due to accidental tagging of the 2.4.1 git tag during CI testing

## API Changes
* Addition of `overwrite` parameter to `Array.create` [#713](https://github.com/TileDB-Inc/TileDB-Py/pull/713)
* Addition of `"ascii"` dtype for `Dim`s [#720](https://github.com/TileDB-Inc/TileDB-Py/pull/720)

## Bug fixes
* Pass `Ctx` to `ArraySchema.load` in `from_pandas` [#709](https://github.com/TileDB-Inc/TileDB-Py/pull/709)
* Give clear error message when attempting to apply `QueryCondition` on dimensions [#722](https://github.com/TileDB-Inc/TileDB-Py/pull/722)
* Do not add string range when querying empty array [#721](https://github.com/TileDB-Inc/TileDB-Py/pull/721)

## Improvements
* String dimension default fix in core [#2436](https://github.com/TileDB-Inc/TileDB/pull/2436) reverts a previous change in which the nonempty domain was passed the to multi-range indexer if unspecified [#712](https://github.com/TileDB-Inc/TileDB-Py/pull/712)

# TileDB-Py 0.10.2 Release Notes

## API Changes
* Deprecate sparse writes to dense arrays [#681](https://github.com/TileDB-Inc/TileDB-Py/pull/681)
* Addition of `Attr.isascii` [#681](https://github.com/TileDB-Inc/TileDB-Py/pull/681)
* Addition of `Ctx.get_stats` and `Query.get_stats` [#698](https://github.com/TileDB-Inc/TileDB-Py/pull/698)

## Improvements
* Added support for `timestamp` argument in `tiledb.from_numpy` [#699](https://github.com/TileDB-Inc/TileDB-Py/pull/699)

# TileDB-Py 0.10.1 Release Notes

## API Changes
* Do not require `domain=(None, None)` for string dimensions [#662](https://github.com/TileDB-Inc/TileDB-Py/pull/662)

## Improvements
* Print a warning about ContextVar bug when running under ipykernel < 6.0. [#665](https://github.com/TileDB-Inc/TileDB-Py/pull/665)
  Please see https://github.com/TileDB-Inc/TileDB-Py/issues/667 for more information.
* `tiledb.Dim` representation now displays `var=True` for dimensions with `bytes` datatype, consistent with `tiledb.Attr` [#669](https://github.com/TileDB-Inc/TileDB-Py/pull/669)

## Bug fixes
* Fix concurrent use of `Array.multi_index` and `.df` by using new instance for each invocation [#672](https://github.com/TileDB-Inc/TileDB-Py/pull/672)
* For attributes, if `var=False` but the bytestring is fixed-width or if `var=True` but the bytestring is variable length, error out [#663](https://github.com/TileDB-Inc/TileDB-Py/pull/663)

# TileDB-Py 0.10.0 Release Notes

## TileDB Embedded updates:
* TileDB-Py 0.10.0 includes TileDB Embedded [TileDB 2.4.0](https://github.com/TileDB-Inc/TileDB/releases/tag/2.4.0) with a number of
  new features and improvements, including:
    - new platform support: Apple M1
    - support for ArraySchema evolution (adding and removing attributes)
    - support for Azure SAS (shared access signature) tokens

## API Changes
* When using `Array.multi_index`, an empty result is returned if the nonempty domain is empty [#656](https://github.com/TileDB-Inc/TileDB-Py/pull/656)
* Addition of `Array.set_query` to read array using a serialized query [#651](https://github.com/TileDB-Inc/TileDB-Py/pull/651)

## Improvements
* Support numeric column names in `from_pandas` by casting to str dtype [#652](https://github.com/TileDB-Inc/TileDB-Py/pull/652)
* New `tiledb.ArraySchemaEvolution` API to add and drop attributes from an existing array [#657](https://github.com/TileDB-Inc/TileDB-Py/pull/657)

## Bug Fixes
* Correct listing of consolidated fragments to vacuum in the Fragment Info API by deprecating `FragmentInfoList.to_vacuum_uri`, `FragmentInfoList.to_vacuum_num`, `FragmentInfo.to_vacuum_uri`, and `FragmentInfo.to_vacuum_num` and replacing with `FragmentInfoList.to_vacuum` [#650](https://github.com/TileDB-Inc/TileDB-Py/pull/650)
* Correct issue where appending `None` to `FilterList` causes segfault by checking the `filter` argument [#653](https://github.com/TileDB-Inc/TileDB-Py/pull/653)

# TileDB-Py 0.9.5 Release Notes

## TileDB Embedded updates:
* TileDB-Py 0.9.5 includes TileDB Embedded [TileDB 2.3.3](https://github.com/TileDB-Inc/TileDB/releases/tag/2.3.3)

## Improvements
* Consolidate `_nonempty_domain_var` into `nonempty_domain` [#632](https://github.com/TileDB-Inc/TileDB-Py/pull/632)
* Support more valid Python syntax for `QueryCondition` statements [#636](https://github.com/TileDB-Inc/TileDB-Py/pull/636)
* Addition of `ascii` dtype to `Attr` allows `QueryCondition` to support var-length strings [#637](https://github.com/TileDB-Inc/TileDB-Py/pull/637)

# TileDB-Py 0.9.4 Release Notes

## Improvements
* Support pickling for arrays in write-mode [#626](https://github.com/TileDB-Inc/TileDB-Py/pull/626)

## Bug Fixes
* Fixed multi-range indexer to default to explicitly pass in the non-empty domain if dimensions are unspecified [#630](https://github.com/TileDB-Inc/TileDB-Py/pull/630)

# TileDB-Py 0.9.3 Release Notes

## Packaging Notes
* Due to a packaging issue released with 0.9.3 (NumPy ABI compatibility with NumPy < 1.20 for Python 3.8), this section is intentionally left blank.

# TileDB-Py 0.9.2 Release Notes

## Packaging Notes
* Fixed release builder ordering issue which led to CRLF line endings in 0.9.1 source distribution.

## API Changes
* Deprecate `Array.timestamp` and replace with `Array.timestamp_range` [#616](https://github.com/TileDB-Inc/TileDB-Py/pull/616)

## Improvements
* Set `ArraySchema.tile_order=None` for Hilbert-ordered arrays [#609](https://github.com/TileDB-Inc/TileDB-Py/pull/609)
* Use CIBW to build release wheels on Linux [#613](https://github.com/TileDB-Inc/TileDB-Py/pull/613)
* Addition of Pickling functionality for `SparseArray` [#618](https://github.com/TileDB-Inc/TileDB-Py/pull/618)

# TileDB-Py 0.9.1 Release Notes

## TileDB Embedded updates:
* TileDB-Py 0.9.1 includes TileDB Embedded [TileDB 2.3.1](https://github.com/TileDB-Inc/TileDB/releases/tag/2.3.1)

## Improvements
* Support passing a timestamp range for consolidation and vacuuming [#603](https://github.com/TileDB-Inc/TileDB-Py/pull/603)

## Bug Fixes
* FragmentInfo API's to_vacuum_uri() function corrected to iterate through `to_vacuum_num` rather than `fragment_num`[#603](https://github.com/TileDB-Inc/TileDB-Py/pull/603)
* Return "NA" For ArraySchema.tile_order if "Hilbert" [#605](https://github.com/TileDB-Inc/TileDB-Py/pull/605)

# TileDB-Py 0.9.0 Release Notes

## TileDB Embedded updates:
* TileDB-Py 0.9.0 includes TileDB Embedded [TileDB 2.3](https://github.com/TileDB-Inc/TileDB/releases/tag/2.3) with a significant
  number of new features and improvements.

## Packaging Notes
* Windows wheels are now built with TileDB Cloud REST support enabled [#541](https://github.com/TileDB-Inc/TileDB-Py/pull/541)

## Improvements
* Addition of `QueryCondition` API to filter query on attributes [#576](https://github.com/TileDB-Inc/TileDB-Py/pull/576)

## Bug Fixes
* Fixed `from_pandas` append error for sparse arrayse: no need to specify 'row_start_idx' [#593](https://github.com/TileDB-Inc/TileDB-Py/pull/593)
* Fixed 'index_dims' kwarg handling for `from_pandas` [#590](https://github.com/TileDB-Inc/TileDB-Py/pull/590)

## API Changes
* `from_dataframe` function has been removed; deprecated in TileDB-Py 0.6 and replaced by `from_pandas`.

---

# TileDB-Py 0.8.11 Release Notes

## Bug fixes
* Fixed incorrect NumPy ABI target in Linux wheels [#590](https://github.com/TileDB-Inc/TileDB-Py/pull/590)
* QueryCondition API will cast condition values to the datatype of the corresponding attribute [#589](https://github.com/TileDB-Inc/TileDB-Py/pull/589)
* QueryCondition API errors out when there are mismatched attributes to `query`'s `attr_cond` and `attrs` arguments [#589](https://github.com/TileDB-Inc/TileDB-Py/pull/589)
* QueryCondition API can now parse negative numbers [#589](https://github.com/TileDB-Inc/TileDB-Py/pull/589)


# TileDB-Py 0.8.10 Release Notes

## Improvements
* Disabled libtiledb Werror compilation argument for from-source builds via setup.py [#574](https://github.com/TileDB-Inc/TileDB-Py/pull/574)
* Relaxed NumPy version requirements for from-source builds via setup.py [#575](https://github.com/TileDB-Inc/TileDB-Py/pull/575)

## Bug fixes
* Fixed FragmentInfoList where context was not being passed to ArraySchema [#573](https://github.com/TileDB-Inc/TileDB-Py/pull/573)
  * Fixed FragmentInfoList where context was not being passed to ArraySchema [#578](https://github.com/TileDB-Inc/TileDB-Py/pull/578)
* Fixed read bug due to large estimated result size [#579](https://github.com/TileDB-Inc/TileDB-Py/pull/579)
* Fixed bug reading nullable attributes due to missing buffer resize [#581](https://github.com/TileDB-Inc/TileDB-Py/pull/581)
* Fixed Python output for `tiledb.stats_dump` [#586](https://github.com/TileDB-Inc/TileDB-Py/pull/586)

# TileDB-Py 0.8.9 Release Notes

## TileDB Embedded updates:
* TileDB-Py 0.8.9 includes TileDB Embedded [TileDB 2.2.9](https://github.com/TileDB-Inc/TileDB/releases/tag/2.2.9)

## Improvements
* Support for iterating over incomplete query results [#548](https://github.com/TileDB-Inc/TileDB-Py/pull/548)
  - This feature provides the capability to consume partial query results with a fixed maximum buffer size
    rather than the the default behavior of resizing buffers and resubmitting to completion.
    Usage example: `examples/incomplete_iteration.py`
    (along with test in: `test_libtiledb.py:test_incomplete_return`)
* Rename FragmentsInfo to FragmentInfoList [#551](https://github.com/TileDB-Inc/TileDB-Py/pull/551)
* Dataframe creation uses Zstd default compression level (-1) [#552](https://github.com/TileDB-Inc/TileDB-Py/pull/552)
* Rename Fragment Info API's `non_empty_domain` attribute to `nonempty_domain` [#553](https://github.com/TileDB-Inc/TileDB-Py/pull/553)
* Added configuration option `py.alloc_max_bytes` to control maximum initial buffer allocation [#557](https://github.com/TileDB-Inc/TileDB-Py/pull/557)

## Bug fixes
* Fixed incorrected error raised in .df[] indexer when pyarrow not installed [#554](https://github.com/TileDB-Inc/TileDB-Py/pull/554)
* Fixed `from_pandas(attr_filters=None, dim_filters=None)` (previously used internal defaults) [#564](https://github.com/TileDB-Inc/TileDB-Py/pull/554)
* Fixed `from_pandas` write bug due to incorrect classification of str/bytes columns [#562](https://github.com/TileDB-Inc/TileDB-Py/pull/562)
* Fix segfault due to mismatched validity num and data buffer sizes [#567](https://github.com/TileDB-Inc/TileDB-Py/pull/567)

# TileDB-Py 0.8.8 Release Notes

## TileDB Embedded updates:
* TileDB-Py 0.8.8 includes TileDB Embedded [TileDB 2.2.8](https://github.com/TileDB-Inc/TileDB/releases/tag/2.2.8)

# TileDB-Py 0.8.7 Release Notes

## Improvements
* ArraySchema support for `cell_order="hilbert"` [#535](https://github.com/TileDB-Inc/TileDB-Py/pull/535)

## Bug fixes
* Fixed regression in `from_pandas` with string-valued index dimensions [#526](https://github.com/TileDB-Inc/TileDB-Py/pull/526)
* Fixed GC lifetime bug in string buffer conversion  [#525](https://github.com/TileDB-Inc/TileDB-Py/pull/526)
* Fixed `FilterList`'s `repr()` method [#528](https://github.com/TileDB-Inc/TileDB-Py/pull/528)

# TileDB-Py 0.8.6 Release Notes

## TileDB Embedded updates:
* TileDB-Py 0.8.6 includes TileDB Embedded [TileDB 2.2.7](https://github.com/TileDB-Inc/TileDB/releases/tag/2.2.7)

## Improvements
* Addition of `VFS()` functions `copy_file()` and `copy_dir()` [#507](https://github.com/TileDB-Inc/TileDB-Py/pull/507)
* Add support in `from_pandas` for storing Pandas extension types as variable-length attributes [#515](https://github.com/TileDB-Inc/TileDB-Py/pull/515)
* Add support for sparse writes to dense arrays [#521](https://github.com/TileDB-Inc/TileDB-Py/pull/521)

## Bug fixes
* Multi-length attributes, regardless of fixed or var-length, do not work query properly with PyArrow enabled due to lack of Arrow List support. When using `.df[]` with PyArrow enabled, we are returning a clear message to the user to use `query(use_pyarrow=False)` [#513](https://github.com/TileDB-Inc/TileDB-Py/pull/513)

# TileDB-Py 0.8.5 Release Notes

## TileDB Embedded updates:
* TileDB-Py 0.8.5 includes TileDB Embedded [TileDB 2.2.6](https://github.com/TileDB-Inc/TileDB/releases/tag/2.2.6)

## Documentation Updates
* Added example reading/writing RGB (multi-component) array [#487](https://github.com/TileDB-Inc/TileDB-Py/pull/487)

## Improvements
* Restore `tiledb.stats_dump` default to `verbose=True` [#491](https://github.com/TileDB-Inc/TileDB-Py/pull/491)
* Remove `non_empty_domain_var()` Fragment Info PyBind11 Function and only use `get_non_empty_domain()` for both fixed and var-length domains [#505](https://github.com/TileDB-Inc/TileDB-Py/pull/505)

# TileDB-Py 0.8.4 Release Notes

## Improvements
* Addition of high-level function `array_fragments()` that returns a `FragmentsInfo` object [#488](https://github.com/TileDB-Inc/TileDB-Py/pull/488)
* Added support for `from_pandas`/`df[]` round-trip of Pandas nullable integer and bool types [#480](https://github.com/TileDB-Inc/TileDB-Py/pull/480)
* Fragment info API example usage now provided at `examples/fragment_info.py` [#479](https://github.com/TileDB-Inc/TileDB-Py/pull/479)
* Fragment info API parameters have been rearranged to match the rest of the TileDB Python API such that the `uri` is provided first and `context`, an optional parameter that defaults to `tiledb.default_ctx()`, is provided second [#479](https://github.com/TileDB-Inc/TileDB-Py/pull/479)

## Bug fixes
* Fix bug in `Attr` to ensure that all Unicode strings are automatically set to `var=True` [#495]https://github.com/TileDB-Inc/TileDB-Py/pull/495
* Fix bug in Array.multi_index slicing bug for sparse array with dimension range including 0 [#482](https://github.com/TileDB-Inc/TileDB-Py/pull/482)

# TileDB-Py 0.8.3 Release Notes

## TileDB Embedded updates:
* TileDB-Py 0.8.3 includes TileDB Embedded [TileDB 2.2.4](https://github.com/TileDB-Inc/TileDB/releases/tag/2.2.4)

## Improvements
* Added `nullable` keyword argument to `Attr` constructor [#474](https://github.com/TileDB-Inc/TileDB-Py/pull/474)

## Bug fixes
* Fix bug in Array.multi_index with slice range including 0 (incorrectly used the nonempty domain as endpoint) [#473](https://github.com/TileDB-Inc/TileDB-Py/pull/473)

# TileDB-Py 0.8.2 Release Notes

## Packaging Notes
* This is a version bump to fix numpy compatibility pinning in the wheel build system.

# TileDB-Py 0.8.1 Release Notes

## TileDB Embedded updates:
* TileDB-Py 0.8.1 includes TileDB Embedded [TileDB 2.2.3](https://github.com/TileDB-Inc/TileDB/releases/tag/2.2.3)

## Packaging Notes
* TileDB-Py 0.8 does not support Python 2.

# TileDB-Py 0.8.0 Release Notes

## TileDB Embedded updates:
* TileDB-Py 0.8.0 includes TileDB Embedded 2.2.2 featuring a number of significant
  improvements in core storage engine functionality. See release notes for
  [TileDB 2.2.1](https://github.com/TileDB-Inc/TileDB/releases/tag/2.2.1) and
  [TileDB 2.2.2](https://github.com/TileDB-Inc/TileDB/releases/tag/2.2.2).

## Packaging Notes
* TileDB-Py 0.8 does not support Python 2.

## Improvements
* Add initial `tiledb.from_parquet` functionality (beta) [[a90d5d9b1b](https://github.com/TileDB-Inc/TileDB-Py/commit/a90d5d9b1b6a39b48090592297fe98a7f33338fb)]
* Preload metadata in .df query path to reduce read latency for remote arrays [[79ab12fcf0](https://github.com/TileDB-Inc/TileDB-Py/commit/79ab12fcf0ede0cbac822392a30ee7640595e93c)]

## Bug fixes
* Update py::dtype usage for compatibility with pybind11 2.6.2 [[9d3d3d3c43](https://github.com/TileDB-Inc/TileDB-Py/commit/9d3d3d3c430fbc058d04773f03ddc63bd47f79e3)]

# TileDB-Py 0.7.7 Release Notes

## Bug fixes
* Cherry-pick commit 9d3d3d3c43 to ix runtime bug in conda packages built against pybind11 2.6.2 [9d3d3d3c430f](https://github.com/TileDB-Inc/TileDB-Py/commit/9d3d3d3c430fbc058d04773f03ddc63bd47f79e3)

# TileDB-Py 0.7.6 Release Notes

## Packaging Notes
* TileDB-Py 0.7.x will be the last version of TileDB-Py supporting Python 2.

## Bug fixes
* Fix read compatibility for empty strings written with 2.1 or 2.2 [#462](https://github.com/TileDB-Inc/TileDB-Py/pull/462)
* Fix #457: make sure to fit automatic tile extent to dim range for date type [#464](https://github.com/TileDB-Inc/TileDB-Py/pull/464)

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
