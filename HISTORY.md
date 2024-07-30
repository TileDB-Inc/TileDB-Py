# Release 0.31.1

## Improvements

* Fix malformed doc str for tiledb.array_schema.ArraySchema in https://github.com/TileDB-Inc/TileDB-Py/pull/2007
* Fix deprecation and test by @kounelisagis in https://github.com/TileDB-Inc/TileDB-Py/pull/2021

## Build system changes

* Add pandas dependency to test group by @dudoslav and @kounelisagis in https://github.com/TileDB-Inc/TileDB-Py/pull/2022
* Define TILEDB_REMOVE_DEPRECATIONS macro for cc and remove deprecated code by @kounelisagis and @dudoslav in https://github.com/TileDB-Inc/TileDB-Py/pull/2023

# Release 0.31.0

* TileDB-Py 0.31.0 includes TileDB Embedded [2.25.0](https://github.com/TileDB-Inc/TileDB/releases/tag/2.25.0)

## Improvements

* Remove deprecated Array.delete_fragments code path by @teo-tsirpanis and @kounelisagis in https://github.com/TileDB-Inc/TileDB-Py/pull/2009
* Fix a typo in an error message by @johnkerl in https://github.com/TileDB-Inc/TileDB-Py/pull/2004
* Support ctx argument in array_exists by @kounelisagis in https://github.com/TileDB-Inc/TileDB-Py/pull/2003
* Move fragment list consolidation API to pybind by @kounelisagis in https://github.com/TileDB-Inc/TileDB-Py/pull/1999

## Build system changes

* Add version pins for pandas 3.0 by @kounelisagis https://github.com/TileDB-Inc/TileDB-Py/pull/2016
* Scikit-build-core build system rework by @dudoslav and @ihnorton in https://github.com/TileDB-Inc/TileDB-Py/pull/1988
* Patches for the build system by @dudoslav, @ihnorton and @kounelisagis in https://github.com/TileDB-Inc/TileDB-Py/pull/2010, https://github.com/TileDB-Inc/TileDB-Py/pull/2011, https://github.com/TileDB-Inc/TileDB-Py/pull/2013, https://github.com/TileDB-Inc/TileDB-Py/pull/2014, https://github.com/TileDB-Inc/TileDB-Py/pull/2018, https://github.com/TileDB-Inc/TileDB-Py/pull/2019, https://github.com/TileDB-Inc/TileDB-Py/pull/2020

# Release 0.30.2

## Packaging Notes

While we currently plan to maintain support for CentOS 7-compatible systems (GLIBC 2.17) through TileDB 2.31, ecosystem and infrastructure updates following the CentOS 7 end-of-life on 30/Jun/2024 may necessitate dropping support earlier. Please contact us if you still use a CentOS 7 (GLIBC 2.17)-like Linux distribution.

## Improvements

* Fix OverflowError: Python int too large to convert to C long by @kounelisagis in https://github.com/TileDB-Inc/TileDB-Py/pull/2000
* Wrap as_built function by @kounelisagis in https://github.com/TileDB-Inc/TileDB-Py/pull/1994
* Fix array.query() incorrectly handling nullables by @kounelisagis in https://github.com/TileDB-Inc/TileDB-Py/pull/1998
* Add offending column when from_pandas -> _get_column_infos fails by @kounelisagis in https://github.com/TileDB-Inc/TileDB-Py/pull/1997

# Release 0.30.1

* TileDB-Py 0.30.1 includes TileDB Embedded [2.24.1](https://github.com/TileDB-Inc/TileDB/releases/tag/2.24.1)

## Improvements

* Document Azure, GCS and local support for VFS.ls_recursive by @kounelisagis in https://github.com/TileDB-Inc/TileDB-Py/pull/1980
* Skip Dask failing test on Windows by @kounelisagis in https://github.com/TileDB-Inc/TileDB-Py/pull/1995

# Release 0.30.0

* TileDB-Py 0.30.0 includes TileDB Embedded [2.24.0](https://github.com/TileDB-Inc/TileDB/releases/tag/2.24.0)

## Improvements

* Add test for blob attribute by @kounelisagis in https://github.com/TileDB-Inc/TileDB-Py/pull/1985
* Deprecate support for [] indexing with floats by @kounelisagis in https://github.com/TileDB-Inc/TileDB-Py/pull/1982
* Fix Query constructor to return error for dense arrays with return_incomplete=True by @kounelisagis in https://github.com/TileDB-Inc/TileDB-Py/pull/1976
* Expose WebP enums by @kounelisagis in https://github.com/TileDB-Inc/TileDB-Py/pull/1974
* Add Array.query in docs and improve docs in general by @kounelisagis in https://github.com/TileDB-Inc/TileDB-Py/pull/1965
* Add support for creating WKB/WKT attributes by @jp-dark in https://github.com/TileDB-Inc/TileDB-Py/pull/1912
* Add wrapping for ls recursive by @kounelisagis in https://github.com/TileDB-Inc/TileDB-Py/pull/1968
* Fix compatibility for delete_fragments by @kounelisagis in https://github.com/TileDB-Inc/TileDB-Py/pull/1966

## Build system changes

* Fix pinning wrong numpy version by @kounelisagis in https://github.com/TileDB-Inc/TileDB-Py/pull/1989
* Remove pin to MSVC toolset version in CI by @teo-tsirpanis in https://github.com/TileDB-Inc/TileDB-Py/pull/1991
* Fix ModuleNotFoundError: No module named 'numpy' on build by @kounelisagis and @ihnorton in https://github.com/TileDB-Inc/TileDB-Py/pull/1979
* Add support for numpy2 by @kounelisagis in https://github.com/TileDB-Inc/TileDB-Py/pull/1969
* Fix syntax error in nightly build workflow by @ihnorton in https://github.com/TileDB-Inc/TileDB-Py/pull/1970
* Set an upper bound for numpy to dodge 2.0 by @sgillies in https://github.com/TileDB-Inc/TileDB-Py/pull/1963

# Release 0.29.1

## Build system changes

* Add numpy upper bound to dodge 2.0 by @kounelisagis in https://github.com/TileDB-Inc/TileDB-Py/pull/1993

# Release 0.29.0

* TileDB-Py 0.29.0 includes TileDB Embedded [2.23.0](https://github.com/TileDB-Inc/TileDB/releases/tag/2.23.0)

## Improvements

* Add wrapping for ls_recursive by @kounelisagis in https://github.com/TileDB-Inc/TileDB-Py/pull/1933
* Migrate away from deprecated TileDB C++ APIs by @kounelisagis in https://github.com/TileDB-Inc/TileDB-Py/pull/1958
* Pybind11 Config should honor prefix for iter by @Shelnutt2 in https://github.com/TileDB-Inc/TileDB-Py/pull/1962
* Fix test skipping by @kounelisagis in https://github.com/TileDB-Inc/TileDB-Py/pull/1957
* Make timestamp overrides optional in tests and add faketime test by @kounelisagis in https://github.com/TileDB-Inc/TileDB-Py/pull/1953
* Wrap tiledb_array_consolidate_fragments from pybind11 by @kounelisagis in https://github.com/TileDB-Inc/TileDB-Py/pull/1948

## Build system changes

* Enable python 3.12 by @dudoslav in https://github.com/TileDB-Inc/TileDB-Py/pull/1959
* Add .vscode to .gitignore by @kounelisagis in https://github.com/TileDB-Inc/TileDB-Py/pull/1952

# Release 0.28.0

## TileDB Embedded updates

* TileDB-Py 0.28.0 includes TileDB Embedded [2.22.0](https://github.com/TileDB-Inc/TileDB/releases/tag/2.22.0)

## Improvements

* Update type signature for VFS::readinto by @kounelisagis in https://github.com/TileDB-Inc/TileDB-Py/pull/1937
* Show enumerated value-types in enum-printer by @kounelisagis in https://github.com/TileDB-Inc/TileDB-Py/pull/1936
* Add wrapping for new consolidation plan API by @kounelisagis in https://github.com/TileDB-Inc/TileDB-Py/pull/1935
* Add test for Group constructor invalid uri object type by @kounelisagis in https://github.com/TileDB-Inc/TileDB-Py/pull/1941
* Update doc for tiledb.consolidate by @ihnorton in https://github.com/TileDB-Inc/TileDB-Py/pull/1946
* Improve documentation of from_numpy function by @kounelisagis in https://github.com/TileDB-Inc/TileDB-Py/pull/1942

## Build system changes

* Exclude .pytest_cache and .hypothesis files by @kounelisagis in https://github.com/TileDB-Inc/TileDB-Py/pull/1932
* Remove modular building option by @kounelisagis in https://github.com/TileDB-Inc/TileDB-Py/pull/1938
* Fix wrong version number for Python API docs by @kounelisagis in https://github.com/TileDB-Inc/TileDB-Py/pull/1947
* Remove conditional code for TileDB < 2.16 by @kounelisagis in https://github.com/TileDB-Inc/TileDB-Py/pull/1949
* Update nightly test target to 2.21 by @ihnorton in https://github.com/TileDB-Inc/TileDB-Py/pull/1950

# Release 0.27.1

## TileDB Embedded updates

* TileDB-Py 0.27.1 includes TileDB Embedded [2.21.1](https://github.com/TileDB-Inc/TileDB/releases/tag/2.21.1)

## Improvements

* Add tests to ensure empty result on query condition for invalid enum. [1882](https://github.com/TileDB-Inc/TileDB-Py/pull/1882)

# Release 0.27.0

* TileDB-Py 0.27.0 includes TileDB Embedded [2.21.0](https://github.com/TileDB-Inc/TileDB/releases/tag/2.21.0)

# Release 0.26.4

## Bug Fixes

* Fix VFS `read`, `seek` with numpy integer sizes. [#1927](https://github.com/TileDB-Inc/TileDB-Py/pull/1927)
* Remove erroneous `_ctx` check for GroupMetadata [#1925](https://github.com/TileDB-Inc/TileDB-Py/pull/1925)

# Release 0.26.3

## Improvements

* Fix vfs readinto when buff is not bytes. [#1915](https://github.com/TileDB-Inc/TileDB-Py/pull/1915)
* Update daily test builds to use single source of truth for libtiledb target versions. [1910](https://github.com/TileDB-Inc/TileDB-Py/pull/1910)
* Remove Python 3.7 CI jobs. [1916](https://github.com/TileDB-Inc/TileDB-Py/pull/1916)

# Release 0.26.2

## Improvements

* Added API support for TileDB aggregates. [#1889](https://github.com/TileDB-Inc/TileDB-Py/pull/1889)
* For compatibility with fsspec and rasterio, `isdir()`, `isfile()`, and `size()` aliases have been added to the `VFS` class. [#1902](https://github.com/TileDB-Inc/TileDB-Py/pull/1902).

# Release 0.26.1

* TileDB-Py 0.26.1 includes TileDB Embedded [2.20.1](https://github.com/TileDB-Inc/TileDB/releases/tag/2.20.1)

# Release 0.26.0

* TileDB-Py 0.26.0 includes TileDB Embedded [2.20.0](https://github.com/TileDB-Inc/TileDB/releases/tag/2.20.0)

## Bug Fixes

* Add safe `Group.__repr__` [#1890](https://github.com/TileDB-Inc/TileDB-Py/pull/1890)
* Use safe repr if ArraySchema was not properly constructed [#1896](https://github.com/TileDB-Inc/TileDB-Py/pull/1896)

## Improvements

* Warn when `os.fork()` is used in the presence of a Tiledb context [#1876](https://github.com/TileDB-Inc/TileDB-Py/pull/1876/files).
* Enable GCS in osx-arm64 wheel builds [#1899](https://github.com/TileDB-Inc/TileDB-Py/pull/1899)

# Release 0.25.0

* TileDB-Py 0.25.0 includes TileDB Embedded [2.19.0](https://github.com/TileDB-Inc/TileDB/releases/tag/2.19.0)

## Improvements

* Fix fill value for complex attributes [1872](https://github.com/TileDB-Inc/TileDB-Py/pull/1872)
* Update current-release nightly target [1873](https://github.com/TileDB-Inc/TileDB-Py/pull/1873)
* Add full check of attribute properties in __eq__ method [1874](https://github.com/TileDB-Inc/TileDB-Py/pull/1874)
* Add all array properties to ArraySchema.__eq__ [1875](https://github.com/TileDB-Inc/TileDB-Py/pull/1875)
* Error out if query condition given empty set [1877](https://github.com/TileDB-Inc/TileDB-Py/pull/1877)

# Release 0.24.0

* TileDB-Py 0.24.0 includes TileDB Embedded [2.18.2](https://github.com/TileDB-Inc/TileDB/releases/tag/2.18.2)

## Improvements

* Release the GIL while consolidation. [#1865](https://github.com/TileDB-Inc/TileDB-Py/pull/1864)
* Add Group Metadata Consolidation and Vacuuming. [#1833](https://github.com/TileDB-Inc/TileDB-Py/pull/1833)
* Fix domain equality check. [#1866](https://github.com/TileDB-Inc/TileDB-Py/pull/1866)
* Fix reading DeltaFilter and DoubleDeltaFilter options for FilterList. [#1862](https://github.com/TileDB-Inc/TileDB-Py/pull/1862)
* Do not flush VFS handles on closing. [#1863](https://github.com/TileDB-Inc/TileDB-Py/pull/1863)
* Add context manager before enter `.array_exists`. [#1868](https://github.com/TileDB-Inc/TileDB-Py/pull/1868) (thanks, new contributor `p4perf4ce`!)

# Release 0.23.4

* TileDB-Py 0.23.4 includes TileDB Embedded [2.17.4](https://github.com/TileDB-Inc/TileDB/releases/tag/2.17.4)

## Improvements

* Add `COMPRESSION_REINTERPRET_DATATYPE` to allowed `FilterOption` [#1855](https://github.com/TileDB-Inc/TileDB-Py/pull/1855)
* Add `filter_name` to `Filter` class [#1856](https://github.com/TileDB-Inc/TileDB-Py/pull/1856)

## Bug Fixes

* Do not use `dtype.kind` in enumeration extend type checking [#1853](https://github.com/TileDB-Inc/TileDB-Py/pull/1853)
* Empty enumerations should be casted to the dtype of the enumeration [#1854](https://github.com/TileDB-Inc/TileDB-Py/pull/1854)
* Correct writing nullable string attributes and all nulled data [#1848](https://github.com/TileDB-Inc/TileDB-Py/pull/1848)
* Pandas 2+ fix: use `pa.schema.with_metadata`, replacing passing metadata to `pa.schema` constructor [#1858](https://github.com/TileDB-Inc/TileDB-Py/pull/1858)

# Release 0.23.3

## Bug Fixes

* Correct `Enumeration.extend` to handle integers, include Booleans, of different sizes [#1850](https://github.com/TileDB-Inc/TileDB-Py/pull/1850)

# Release 0.23.2

* TileDB-Py 0.23.2 includes TileDB Embedded [2.17.3](https://github.com/TileDB-Inc/TileDB/releases/tag/2.17.3)

## Improvements

* Add support for enumeration extension API. [#1847](https://github.com/TileDB-Inc/TileDB-Py/pull/1847)
* Support new set membership query condition. [#1837](https://github.com/TileDB-Inc/TileDB-Py/pull/1837)
* Create `ArraySchemaEvolution` for new operation. [#1839](https://github.com/TileDB-Inc/TileDB-Py/pull/1839)
* Add sparse dimension label example. [#1843](https://github.com/TileDB-Inc/TileDB-Py/pull/1843)

# Release 0.23.1

* TileDB-Py 0.23.1 includes TileDB Embedded [2.17.1](https://github.com/TileDB-Inc/TileDB/releases/tag/2.17.1)

# Release 0.23.0

* TileDB-Py 0.23.0 includes TileDB Embedded [2.17.0](https://github.com/TileDB-Inc/TileDB/releases/tag/2.17.0)

## Improvements

* Support for "enumerated datatypes" (aka categoricals or factors). [#1790](https://github.com/TileDB-Inc/TileDB-Py/pull/1790)
* Introduce `Array.read_subarray` and `Array.write_subarray` APIs. [#1824](https://github.com/TileDB-Inc/TileDB-Py/pull/1824)
* Avoid importing Pandas until we actually use it. [#1825](https://github.com/TileDB-Inc/TileDB-Py/pull/1825)
* Make VFS accept path-like objects to refer to files. [#1818](https://github.com/TileDB-Inc/TileDB-Py/pull/1818)

## Bug Fixes

* Use object equality check in buffer conversion, fixes state serialization bug in distributed use-case. [#1822](https://github.com/TileDB-Inc/TileDB-Py/pull/1822)

# Release 0.22.3

* TileDB-Py 0.22.3 includes TileDB Embedded [2.16.3](https://github.com/TileDB-Inc/TileDB/releases/tag/2.16.3)

## Improvements

* Wrap Delta filter. [#1710](https://github.com/TileDB-Inc/TileDB-Py/pull/1710)

# Release 0.22.2

* TileDB-Py 0.22.2 includes TileDB Embedded [2.16.2](https://github.com/TileDB-Inc/TileDB/releases/tag/2.16.2)

# Release 0.22.1

* TileDB-Py 0.22.1 includes TileDB Embedded [2.16.1](https://github.com/TileDB-Inc/TileDB/releases/tag/2.16.1)

# Release 0.22.0

* TileDB-Py 0.22.0 includes TileDB Embedded [2.16.0](https://github.com/TileDB-Inc/TileDB/releases/tag/2.16.0)

## Improvements

* Added support for variable-length dimension label reads [#1802](https://github.com/TileDB-Inc/TileDB-Py/pull/1802)

## Bug Fixes

* Fix online help typo, and clarify. [#1803](https://github.com/TileDB-Inc/TileDB-Py/pull/1803)
* Fix bad memory access for dimension label tile. [#1804](https://github.com/TileDB-Inc/TileDB-Py/pull/1804)

# Release 0.21.6

## Bug Fixes

* Fix group.close to work on 2.16. [#1793](https://github.com/TileDB-Inc/TileDB-Py/pull/1793)
* Fix sc-30787, VFS S3 write failure. [#1794](https://github.com/TileDB-Inc/TileDB-Py/pull/1794)

# Release 0.21.5

## TileDB Embedded updates

* TileDB-Py 0.21.5 includes TileDB Embedded [2.15.4](https://github.com/TileDB-Inc/TileDB/releases/tag/2.15.4)

## Bug Fixes

* Handle empty string passed to Query condition [#1774](https://github.com/TileDB-Inc/TileDB-Py/pull/1774)

# Release 0.21.4

## TileDB Embedded updates

* TileDB-Py 0.21.4 includes TileDB Embedded [2.15.3](https://github.com/TileDB-Inc/TileDB/releases/tag/2.15.3)

# Release 0.21.3

## TileDB Embedded updates

* TileDB-Py 0.21.3 includes TileDB Embedded [2.15.2](https://github.com/TileDB-Inc/TileDB/releases/tag/2.15.2)

## Improvements

* Add support for Group.delete; fixes sc-28030. [#1754](https://github.com/TileDB-Inc/TileDB-Py/pull/1754)

## Bug Fixes

* Fix sc-27374: default order mapping fallback. [#1736](https://github.com/TileDB-Inc/TileDB-Py/pull/1736)
* Fix for array_exists hiding errors; fixes SC-27849. [#1754](https://github.com/TileDB-Inc/TileDB-Py/pull/1754)

# Release 0.21.2

## TileDB Embedded updates

* TileDB-Py 0.21.2 includes TileDB Embedded [2.15.1](https://github.com/TileDB-Inc/TileDB/releases/tag/2.15.1)

## Improvements

* Addition of `Group(config=...)` and `Group.set_config` [#1715](https://github.com/TileDB-Inc/TileDB-Py/pull/1715)

# Release 0.21.1

*0.21.0 tag was invalid and thus deleted before PyPI release.*

## TileDB Embedded updates
* TileDB-Py 0.21.0 includes TileDB Embedded [2.15.0](https://github.com/TileDB-Inc/TileDB/releases/tag/2.15.0)

## Improvements
* Move `ArraySchema` from Cython to pure Python [#1340](https://github.com/TileDB-Inc/TileDB-Py/pull/1340)

## Bug Fixes
* Correct `Attr.fill` for UTF-8 [#1594](https://github.com/TileDB-Inc/TileDB-Py/pull/1594)

# Release 0.20.0

## TileDB Embedded updates
* TileDB-Py 0.20.0 includes TileDB Embedded [TileDB 2.14.0](https://github.com/TileDB-Inc/TileDB/releases/tag/2.14.0)

## Bug Fixes

* Introduce safe repr for classes for `Filter`, `FilterList`, `Dim`, `Domain`, and `Attr`. [#1545](https://github.com/TileDB-Inc/TileDB-Py/pull/1545), [#1555](https://github.com/TileDB-Inc/TileDB-Py/pull/1555)

# Release 0.19.1

## TileDB Embedded updates:
* TileDB-Py 0.19.1 includes TileDB Embedded [TileDB 2.13.1](https://github.com/TileDB-Inc/TileDB/releases/tag/2.13.1)

## Improvements
* Move `Dim` and `Domain` from Cython to pure Python [#1327](https://github.com/TileDB-Inc/TileDB-Py/pull/1327)

## Bug Fixes
* Ensure NumPy array matches array schema dimensions for dense writes [#1514](https://github.com/TileDB-Inc/TileDB-Py/pull/1514)

# Release 0.19.0

## Packaging Notes
* Added support for Python 3.11

## TileDB Embedded updates:
* TileDB-Py 0.19.0 includes TileDB Embedded [TileDB 2.13.0](https://github.com/TileDB-Inc/TileDB/releases/tag/2.13.0)

## Deprecations
* `FragmentInfoList.non_empty_domain` deprecated for `FragmentInfoList.nonempty_domain`
* `FragmentInfoList.to_vacuum_num` deprecated for `len(FragmentInfoList.to_vacuum)`
* `FragmentInfoList.to_vacuum_uri` deprecated for `FragmentInfoList.to_vacuum`
* `FragmentInfoList.dense` deprecated for `not FragmentInfoList.dense`
* `FragmentInfo.non_empty_domain` deprecated for `FragmentInfo.nonempty_domain`
* `FragmentInfo.to_vacuum_num` deprecated for `len(FragmentInfo.to_vacuum)`
* `FragmentInfo.to_vacuum_uri` deprecated for `FragmentInfo.to_vacuum`
* `FragmentInfo.dense` deprecated for `not FragmentInfo.dense`
* `FragmentsInfo` deprecated for `FragmentInfoList`
* `tiledb.delete_fragments` deprecated for `Array.delete_fragments`
* `Array.timestamp` deprecated for `Array.timestamp_range`
* `Array.coords_dtype` deprecated with no replacement; combined coords have been removed from libtiledb
* `Array.timestamp` deprecated for `Array.timestamp_range`
* `Array.query(attr_cond=...)` deprecated for `Array.query(cond=...)`
* `Array.query(cond=tiledb.QueryCondition('expression'))` deprecated for `Array.query(cond='expression')`

## API Changes
* Add support for `WebpFilter` [#1395](https://github.com/TileDB-Inc/TileDB-Py/pull/1395)
* Support Boolean types for query conditions [#1432](https://github.com/TileDB-Inc/TileDB-Py/pull/1432)
* Support for partial consolidation using a list of fragment URIs [#1431](https://github.com/TileDB-Inc/TileDB-Py/pull/1431)
* Addition of `ArraySchemaEvolution.timestamp` [#1480](https://github.com/TileDB-Inc/TileDB-Py/pull/1480)
* Addition of `ArraySchema.has_dim` [#1430](https://github.com/TileDB-Inc/TileDB-Py/pull/1430)
* Addition of `Array.delete_array` [#1428](https://github.com/TileDB-Inc/TileDB-Py/pull/1428)

## Bug Fixes
* Fix issue where queries in delete mode error out on arrays with string dimensions [#1473](https://github.com/TileDB-Inc/TileDB-Py/pull/1473)
* Fix representation of nullable integers in dataframe when using PyArrow path [#1439](https://github.com/TileDB-Inc/TileDB-Py/pull/1439)
* Check for uninitialized query state after submit and error out if uninitialized [#1483](https://github.com/TileDB-Inc/TileDB-Py/pull/1483)

# Release 0.18.3

## Packaging Notes
* Linux wheels now built on `manylinux2014`; previously built on `manylinux2010`
* Windows wheels NOT AVAILABLE for this release

## TileDB Embedded updates:
* TileDB-Py 0.18.3 includes TileDB Embedded [TileDB 2.12.3](https://github.com/TileDB-Inc/TileDB/releases/tag/2.12.3)

## Improvements
* Move `from_numpy` out of Cython into pure Python [#1436](https://github.com/TileDB-Inc/TileDB-Py/pull/1436)

## Bug Fixes
* Fix `.df` and `.multi_index` always returning attributes applied in query conditions [#1433](https://github.com/TileDB-Inc/TileDB-Py/pull/1433)

# Release 0.18.2

## TileDB Embedded updates:
* TileDB-Py 0.18.2 includes TileDB Embedded [TileDB 2.12.2](https://github.com/TileDB-Inc/TileDB/releases/tag/2.12.2)

# Release 0.18.1

## TileDB Embedded updates:
* TileDB-Py 0.18.1 includes TileDB Embedded [TileDB 2.12.1](https://github.com/TileDB-Inc/TileDB/releases/tag/2.12.1)

## Improvements
* Reintroduce moving `Attr` from Cython to pure Python [#1411](https://github.com/TileDB-Inc/TileDB-Py/pull/1411)

## Bug Fixes
* Properly handle whitespaces in a query condition [#1398](https://github.com/TileDB-Inc/TileDB-Py/pull/1398)

# Release 0.18.0

## TileDB Embedded updates:
* TileDB-Py 0.18.0 includes TileDB Embedded [TileDB 2.12.0](https://github.com/TileDB-Inc/TileDB/releases/tag/2.12.0)

## API Changes
* Changes to query conditions [#1341](https://github.com/TileDB-Inc/TileDB-Py/pull/1341)
    * Support query conditions on sparse dimensions
    * Deprecate `attr_cond` in favor of `cond`
    * Deprecate passing `tiledb.QueryCondition` to `cond` in favor of passing string directly
* Add support for `XORFilter` [#1294](https://github.com/TileDB-Inc/TileDB-Py/pull/1294)
* Addition of `Array.delete_fragments`; deprecate `tiledb.delete_fragments` [#1329](https://github.com/TileDB-Inc/TileDB-Py/pull/1329)
* Array and Group metadata now store bytes as `TILEDB_BLOB` [#1384](https://github.com/TileDB-Inc/TileDB-Py/pull/1384)
* Addition of `{Array,Group}.metadata.dump()` [#1384](https://github.com/TileDB-Inc/TileDB-Py/pull/1384)
* Addition of `Group.is_relative` to check if the URI component of a group member is relative [#1386](https://github.com/TileDB-Inc/TileDB-Py/pull/1386)
* Addition of query deletes to delete data that satisifies a given query condition [#1309](https://github.com/TileDB-Inc/TileDB-Py/pull/1309)
* Addition of `FileIO.readinto` [#1389](https://github.com/TileDB-Inc/TileDB-Py/pull/1389)

## Improvements
* Addition of Utility Function `get_last_ctx_err_str()` for C API [#1351](https://github.com/TileDB-Inc/TileDB-Py/pull/1351)
* Move `Context` and `Config` from Cython to pure Python [#1379](https://github.com/TileDB-Inc/TileDB-Py/pull/1379)

# TileDB-Py 0.17.6 Release Notes

## Bug Fixes
* Correct writing empty/null strings to array. `tiledb.main.array_to_buffer` needs to resize data buffer at the end of `convert_unicode`; otherwise, last cell will be store with trailing nulls chars [#1339](https://github.com/TileDB-Inc/TileDB-Py/pull/1339)
* Revert [#1326](https://github.com/TileDB-Inc/TileDB-Py/pull/1326) due to issues with `Context` lifetime with in multiprocess settings [#1372](https://github.com/TileDB-Inc/TileDB-Py/pull/1372)

# TileDB-Py 0.17.5 Release Notes

## Improvements
* Move `Attr` from Cython to pure Python [#1326](https://github.com/TileDB-Inc/TileDB-Py/pull/1326)
* Move `Domain` and `Dim` from Cython to pure Python [#1327](https://github.com/TileDB-Inc/TileDB-Py/pull/1327)

## API Changes
* Permit true-ASCII attributes in non-from-pandas dataframes [#1337](https://github.com/TileDB-Inc/TileDB-Py/pull/1337)
* Addition of `Array.upgrade_version` to upgrade array to latest version [#1334](https://github.com/TileDB-Inc/TileDB-Py/pull/1334)
* Attributes in query conditions no longer need to be passed to `Array.query`'s `attr` arg [#1333](https://github.com/TileDB-Inc/TileDB-Py/pull/1333)
* `ArraySchemaEvolution` checks context's last error for error message [#1335](https://github.com/TileDB-Inc/TileDB-Py/pull/1335)

# TileDB-Py 0.17.4 Release Notes

## TileDB Embedded updates:
* TileDB-Py 0.17.4 includes TileDB Embedded [TileDB 2.11.3](https://github.com/TileDB-Inc/TileDB/releases/tag/2.11.3)

## API Changes
* Addition of `FloatScaleFilter` [#1195](https://github.com/TileDB-Inc/TileDB-Py/pull/1195)
* Addition of `d` mode for arrays to delete data that satisfies a given query condition [#1309](https://github.com/TileDB-Inc/TileDB-Py/pull/1309)

## Misc Updates
* Wheels are minimally supported for macOS 10.15 Catalina [#1275](https://github.com/TileDB-Inc/TileDB-Py/pull/1275)

# TileDB-Py 0.17.3 Release Notes

## TileDB Embedded updates:
* TileDB-Py 0.17.3 includes TileDB Embedded [TileDB 2.11.2](https://github.com/TileDB-Inc/TileDB/releases/tag/2.11.2)

## API Changes
* Add ability to pass shape tuple to empty_like [#1316](https://github.com/TileDB-Inc/TileDB-Py/pull/1316)
* Support retrieving MBRs of var-sized dimensions [#1311](https://github.com/TileDB-Inc/TileDB-Py/pull/1311)

## Misc Updates
* Wheels will no longer be supported for macOS 10.15 Catalina; the minimum supported macOS version is now 11 Big Sur [#1300](https://github.com/TileDB-Inc/TileDB-Py/pull/1300)
* Wheels will no longer supported for Python 3.6 [#1300](https://github.com/TileDB-Inc/TileDB-Py/pull/1300)


# TileDB-Py 0.17.2 Release Notes

## TileDB Embedded updates:
* TileDB-Py 0.17.2 includes TileDB Embedded [TileDB 2.11.1](https://github.com/TileDB-Inc/TileDB/releases/tag/2.11.1)

## Bug Fixes
* Fix issue where querying an array with a Boolean type when `arrow=True`, but is unselected in `.query(attr=...)`, results in an error `pyarrow.lib.ArrowInvalid: Invalid column index to set field.` [#1291](https://github.com/TileDB-Inc/TileDB-Py/pull/1291)
* Use Arrow type fixed-width binary ("w:") for non-variable TILEDB_CHAR [#1286](https://github.com/TileDB-Inc/TileDB-Py/pull/1286)

# TileDB-Py 0.17.1 Release Notes

## API Changes
* Support `datetime64` for `QueryCondition` [#1279](https://github.com/TileDB-Inc/TileDB-Py/pull/1279)

# TileDB-Py 0.17.0 Release Notes

## TileDB Embedded updates:
* TileDB-Py 0.17.0 includes TileDB Embedded [TileDB 2.11.0](https://github.com/TileDB-Inc/TileDB/releases/tag/2.11.0)

# TileDB-Py 0.16.5 Release Notes

## TileDB Embedded updates:
* TileDB-Py 0.16.5 includes TileDB Embedded [TileDB 2.10.4](https://github.com/TileDB-Inc/TileDB/releases/tag/2.10.4)

# TileDB-Py 0.16.4 Release Notes

## TileDB Embedded updates:
* TileDB-Py 0.16.4 includes TileDB Embedded [TileDB 2.10.3](https://github.com/TileDB-Inc/TileDB/releases/tag/2.10.3)

## Improvements
* `setup.py` revert back to retrieving core version by using `ctypes` by parsing `tiledb_version.h`; the tiledb shared object lib now returns back a full path [#1226](https://github.com/TileDB-Inc/TileDB-Py/pull/1226)
* Update minimum required cmake version to =>3.23; required for building `libtiledb` [#1260](https://github.com/TileDB-Inc/TileDB-Py/pull/1260)

## API Changes
* Addition of `in` operator for `QueryCondition` [#1214](https://github.com/TileDB-Inc/TileDB-Py/pull/1214)
* Revert the regular indexer `[:]` to return entire array rather than nonempty domain in order to maintain NumPy semantics [#1261](https://github.com/TileDB-Inc/TileDB-Py/pull/1261)

## Bug Fixes
* Deprecate `Filestore.import_uri` in lieu of `Filestore.copy_from` [#1226](https://github.com/TileDB-Inc/TileDB-Py/pull/1226)

# TileDB-Py 0.16.3 Release Notes

## Packaging Notes
* This removes `import tkinter` from `test_libtiledb.py` which was preventing the conda package from building properly

# TileDB-Py 0.16.2 Release Notes

## TileDB Embedded updates:
* TileDB-Py 0.16.2 includes TileDB Embedded [TileDB 2.10.2](https://github.com/TileDB-Inc/TileDB/releases/tag/2.10.2)

## Improvements
* `setup.py` retrieves core version by using `ctypes` to call `tiledb_version` rather than parsing `tiledb_version.h` [#1191](https://github.com/TileDB-Inc/TileDB-Py/pull/1191)

## Bug Fixes
* Set nonempty domain to `(None, None)` for empty string [#1182](https://github.com/TileDB-Inc/TileDB-Py/pull/1182)

## API Changes
* Support `QueryCondition` for dense arrays [#1198](https://github.com/TileDB-Inc/TileDB-Py/pull/1198)
* Querying dense array with `[:]` returns shape that matches nonempty domain, consistent with `.df[:]` and `.multi_index[:]` [#1199](https://github.com/TileDB-Inc/TileDB-Py/pull/1199)
* Addition of `from_numpy` support for `mode={ingest,schema_only,append}` [#1185](https://github.com/TileDB-Inc/TileDB-Py/pull/1185)

# TileDB-Py 0.16.1 Release Notes

## TileDB Embedded updates:
* TileDB-Py 0.16.1 includes TileDB Embedded [TileDB 2.10.1](https://github.com/TileDB-Inc/TileDB/releases/tag/2.10.1)

# TileDB-Py 0.16.0 Release Notes

## TileDB Embedded updates:
* TileDB-Py 0.16.0 includes TileDB Embedded [TileDB 2.10.0](https://github.com/TileDB-Inc/TileDB/releases/tag/2.10.0)

## API Changes
* Addition of `Filestore` API [#1070](https://github.com/TileDB-Inc/TileDB-Py/pull/1070)
* Use `bool` instead of `uint8` for Boolean dtype in `dataframe_.py` [#1154](https://github.com/TileDB-Inc/TileDB-Py/pull/1154)
* Support `QueryCondition` OR operator [#1146](https://github.com/TileDB-Inc/TileDB-Py/pull/1146)

# TileDB-Py 0.15.6 Release Notes

## TileDB Embedded updates:
* TileDB-Py 0.15.6 includes TileDB Embedded [TileDB 2.9.5](https://github.com/TileDB-Inc/TileDB/releases/tag/2.9.5)

# TileDB-Py 0.15.5 Release Notes

## TileDB Embedded updates:
* TileDB-Py 0.15.5 includes TileDB Embedded [TileDB 2.9.4](https://github.com/TileDB-Inc/TileDB/releases/tag/2.9.4)

## API Changes
* Support `TILEDB_BLOB` dtype [#1159](https://github.com/TileDB-Inc/TileDB-Py/pull/1159)

## Bug Fixes
* Fix error where passing a `Context` to `Group` would segfault intermittenly [#1165](https://github.com/TileDB-Inc/TileDB-Py/pull/1165)
* Correct Boolean values when `use_arrow=True` [#1167](https://github.com/TileDB-Inc/TileDB-Py/pull/1167)

# TileDB-Py 0.15.4 Release Notes

## Packaging Notes
* Due to a packaging error, the wheels for 0.15.4 should not be used.

# TileDB-Py 0.15.3 Release Notes

## TileDB Embedded updates:
* TileDB-Py 0.15.3 includes TileDB Embedded [TileDB 2.9.3](https://github.com/TileDB-Inc/TileDB/releases/tag/2.9.3)

# TileDB-Py 0.15.2 Release Notes

## TileDB Embedded updates:
* TileDB-Py 0.15.2 includes TileDB Embedded [TileDB 2.9.2](https://github.com/TileDB-Inc/TileDB/releases/tag/2.9.2)

## Improvements
* Refactor `MultiRangeIndexer` & `DataFrameIndexer`: addition of ABC `_BaseIndexer` with virtual method `_run_query` and generator `_BaseIndexer.__iter__`; remove `_iter_state`; and fix bugs related to incomplete queries [#1134](https://github.com/TileDB-Inc/TileDB-Py/pull/1134)

## Bug Fixes
* Fix race condition in `{Dense,Sparse}Array.__new__` [#1096](https://github.com/TileDB-Inc/TileDB-Py/pull/1096)
* Correcting `stats_dump` issues: Python stats now also in JSON form if `json=True`, resolve name mangling of `json` argument and `json` module, and pulling "timer" and "counter" stats from `stats_json_core` for `libtiledb`>=2.3  [#1140](https://github.com/TileDB-Inc/TileDB-Py/pull/1140)

## API Changes
* Addition of `tiledb.DictionaryFilter` [#1074](https://github.com/TileDB-Inc/TileDB-Py/pull/1074)
* Add support for `Datatype::TILEDB_BOOL` [#1110](https://github.com/TileDB-Inc/TileDB-Py/pull/1110)
* Addition of `Group.__contains__` to check if member with given name is in Group [#1125](https://github.com/TileDB-Inc/TileDB-Py/pull/1125)
* Support with-statement for `Group`s [#1124](https://github.com/TileDB-Inc/TileDB-Py/pull/1124)
* Addition of `keys`, `values`, and `items` to `Group.meta` [#1123](https://github.com/TileDB-Inc/TileDB-Py/pull/1123)
* `Group.member` also returns name if given [#1141](https://github.com/TileDB-Inc/TileDB-Py/pull/1141)

# TileDB-Py 0.15.1 Release Notes

## TileDB Embedded updates:
* TileDB-Py 0.15.1 includes TileDB Embedded [TileDB 2.9.1](https://github.com/TileDB-Inc/TileDB/releases/tag/2.9.1)

# TileDB-Py 0.15.0 Release Notes

## TileDB Embedded updates:
* TileDB-Py 0.15.0 includes TileDB Embedded [TileDB 2.9.0](https://github.com/TileDB-Inc/TileDB/releases/tag/2.9.0)

## Misc Updates
* Wheels will no longer be supported for macOS 10.14 Mojave; the minimum supported macOS version is now 10.15 Catalina [#1080](https://github.com/TileDB-Inc/TileDB-Py/pull/1080)

# TileDB-Py 0.14.5 Release Notes

## TileDB Embedded updates:
* TileDB-Py 0.14.5 includes TileDB Embedded [TileDB 2.8.3](https://github.com/TileDB-Inc/TileDB/releases/tag/2.8.3)

# TileDB-Py 0.14.4 Release Notes

## Misc Updates
* Update `MACOSX_DEPLOYMENT_TARGET` from 10.14 to 10.15 [#1080](https://github.com/TileDB-Inc/TileDB-Py/pull/1080)

## Bug Fixes
* Correct handling of Arrow cell count with all empty result [#1082](https://github.com/TileDB-Inc/TileDB-Py/pull/1082)

# TileDB-Py 0.14.3 Release Notes

## Improvements
* Refactor display of TileDB objects in Jupyter notebooks to be more readable [#1049](https://github.com/TileDB-Inc/TileDB-Py/pull/1049)
* Improve documentation for `Filter`, `FilterList`, `VFS`, `FileIO`, `Group`, and  `QueryCondition` [#1043](https://github.com/TileDB-Inc/TileDB-Py/pull/1043), [#1058](https://github.com/TileDB-Inc/TileDB-Py/pull/1058)

## Bug Fixes
* `Dim.shape` correctly errors out if type is not integer or datetime [#1055](https://github.com/TileDB-Inc/TileDB-Py/pull/1055)
* Correctly check dtypes in `from_pandas` for supported versions of NumPy <1.20 [#1054](https://github.com/TileDB-Inc/TileDB-Py/pull/1054)
* Fix Arrow Table lifetime issues when using`.query(return_arrow=True)` [#1056](https://github.com/TileDB-Inc/TileDB-Py/pull/1056)

# TileDB-Py 0.14.2 Release Notes

## TileDB Embedded updates:
* TileDB-Py 0.14.2 includes TileDB Embedded [TileDB 2.8.2](https://github.com/TileDB-Inc/TileDB/releases/tag/2.8.2)

## Improvements
* Add `Group` and `Object` to docs [#1040](https://github.com/TileDB-Inc/TileDB-Py/pull/1040)

## Bug Fixes
* Correct `Group.__repr__` to call correct `_dump` function [#1040](https://github.com/TileDB-Inc/TileDB-Py/pull/1040)
* Check type of `ctx` in `from_pandas` and `from_csv` [#1042](https://github.com/TileDB-Inc/TileDB-Py/pull/1042)
* Only allow use of `.df` indexer for `.query(return_arrow=True)`; error out with meaningful error message otherwise [#1045](https://github.com/TileDB-Inc/TileDB-Py/pull/1045)

# TileDB-Py 0.14.1 Release Notes

## TileDB Embedded updates:
* TileDB-Py 0.14.1 includes TileDB Embedded [TileDB 2.8.1](https://github.com/TileDB-Inc/TileDB/releases/tag/2.8.1)

# TileDB-Py 0.14.0 Release Notes

## TileDB Embedded updates:
* TileDB-Py 0.14.0 includes TileDB Embedded [TileDB 2.8.0](https://github.com/TileDB-Inc/TileDB/releases/tag/2.8.0)

## API Changes
* Addition of `Group` and `Object` classes to support improved groups [#1022](https://github.com/TileDB-Inc/TileDB-Py/pull/1022)

# TileDB-Py 0.13.3 Release Notes

## TileDB Embedded updates:
* The Python 3.10 / manylinux2014 wheels for TileDB-Py 0.13.3 include TileDB Embedded [TileDB 2.7.2](https://github.com/TileDB-Inc/TileDB/releases/tag/2.7.2) built with GCS support

## Improvements
* Move `VFS`, `FileIO`, and `FileHandle` classes from Cython to Pybind11 [#934](https://github.com/TileDB-Inc/TileDB-Py/pull/934)

# TileDB-Py 0.13.2 Release Notes

## TileDB Embedded updates:
* TileDB-Py 0.13.2 includes TileDB Embedded [TileDB 2.7.2](https://github.com/TileDB-Inc/TileDB/releases/tag/2.7.2)

## Improvements
* Move `FilterList` and `Filter` classes from Cython to Pybind11 [#921](https://github.com/TileDB-Inc/TileDB-Py/pull/921)

## Bug Fixes
* Fix default validity for write to nullable attribute [#994](https://github.com/TileDB-Inc/TileDB-Py/pull/994)
* Reduce query time for dense var-length arrays by including extra offset element in initial buffer allocation [#1005](https://github.com/TileDB-Inc/TileDB-Py/pull/1005)
* Fix round-trippable repr for dimension tile [#998](https://github.com/TileDB-Inc/TileDB-Py/pull/998)

## API Changes
* Addition of `ArraySchema.version` to get version of array schema [#949](https://github.com/TileDB-Inc/TileDB-Py/pull/949)
* Deprecate `coords_filters` from `ArraySchema` [#993](https://github.com/TileDB-Inc/TileDB-Py/pull/993)
* Allow setting `ascii` in `column_type` for `from_pandas`/`from_csv` [#999](https://github.com/TileDB-Inc/TileDB-Py/pull/999)

# TileDB-Py 0.13.1 Release Notes

## TileDB Embedded updates:
* TileDB-Py 0.13.1 includes TileDB Embedded [TileDB 2.7.1](https://github.com/TileDB-Inc/TileDB/releases/tag/2.7.1)

# TileDB-Py 0.13.0 Release Notes

## TileDB Embedded updates:
* TileDB-Py 0.13.0 includes TileDB Embedded [TileDB 2.7.0](https://github.com/TileDB-Inc/TileDB/releases/tag/2.7.0)

## API Changes
* Deprecate partial vacuuming [#930](https://github.com/TileDB-Inc/TileDB-Py/pull/930)
* Default `from_csv` to use `filter=ZstdFilter()` if not specified for `Attr` or `Dim` [#937](https://github.com/TileDB-Inc/TileDB-Py/pull/937)

# TileDB-Py 0.12.4 Release Notes

## TileDB Embedded updates:
* TileDB-Py 0.12.4 includes TileDB Embedded [TileDB 2.6.4](https://github.com/TileDB-Inc/TileDB/releases/tag/2.6.4)

# TileDB-Py 0.12.3 Release Notes

## TileDB Embedded updates:
* TileDB-Py 0.12.3 includes TileDB Embedded [TileDB 2.6.3](https://github.com/TileDB-Inc/TileDB/releases/tag/2.6.3)

## Bug Fixes
* Properly initalize query in order to retrieve estimate results [#920](https://github.com/TileDB-Inc/TileDB-Py/pull/920)
* Enable building with serialization disabled [#924](https://github.com/TileDB-Inc/TileDB-Py/pull/924)
* Do not print out `FragmentInfo_frags` for `repr` [#925](https://github.com/TileDB-Inc/TileDB-Py/pull/925)
* Error out with `IndexError` when attempting to use a step in the regular indexer [#911](https://github.com/TileDB-Inc/TileDB-Py/pull/911)

# TileDB-Py 0.12.2 Release Notes

## TileDB Embedded updates:
* TileDB-Py 0.12.2 includes TileDB Embedded [TileDB 2.6.2](https://github.com/TileDB-Inc/TileDB/releases/tag/2.6.2)

## API Changes
* Addition of `ArraySchema.validity_filters` [#898](https://github.com/TileDB-Inc/TileDB-Py/pull/898)

# TileDB-Py 0.12.1 Release Notes

## TileDB Embedded updates:
* TileDB-Py 0.12.1 includes TileDB Embedded [TileDB 2.6.1](https://github.com/TileDB-Inc/TileDB/releases/tag/2.6.1)
## Bug fixes
* Cast 'dim`'s dtype in `Domain` to `str` prior to applying `html.escape` [#883](https://github.com/TileDB-Inc/TileDB-Py/pull/883)
* Support attributes with spaces in `QueryCondition` by casting with attr(); values may be casted with val() [#886](https://github.com/TileDB-Inc/TileDB-Py/pull/886)

# TileDB-Py 0.12.0 Release Notes

## TileDB Embedded updates:
* TileDB-Py 0.12.0 includes TileDB Embedded [TileDB 2.6.0](https://github.com/TileDB-Inc/TileDB/releases/tag/2.6.0)

## API Changes
* Allow writing to dimension-only array (zero attributes) by using assignment to `None`, for example: `A[coords] = None` (given `A: tiledb.Array`) [#854](https://github.com/TileDB-Inc/TileDB-Py/pull/854)
* Remove repeating header names for `attr` when displaying `ArraySchema` in Jupyter Notebooks [#856](https://github.com/TileDB-Inc/TileDB-Py/pull/856)
* `tiledb.VFS.open` returns `FileIO` object; no longer returns `FileHandle` [#802](https://github.com/TileDB-Inc/TileDB-Py/pull/802)
* Addition of `tiledb.copy_fragments_to_existing_array` [#864](https://github.com/TileDB-Inc/TileDB-Py/pull/864)

## Bug fixes
* HTML escape strings for `Dim` and `Attr`'s `name` and `dtype` [#856](https://github.com/TileDB-Inc/TileDB-Py/pull/856)
* Fix attribute view for multi-indexer [#866](https://github.com/TileDB-Inc/TileDB-Py/pull/866)

## Improvements
* Metadata-related API calls are now 'nogil' [#867](https://github.com/TileDB-Inc/TileDB-Py/pull/867)

# TileDB-Py 0.11.5 Release Notes

* Added missing dependency on [`packaging`](https://pypi.org/project/packaging/) in requirements.txt [#852](https://github.com/TileDB-Inc/TileDB-Py/pull/852)

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

## Bug fixes
* Support dict parameter for 'config' argument to VFS constructor [#805](https://github.com/TileDB-Inc/TileDB-Py/pull/805)
* Correct libtiledb version checking for Fragment Info API getters' MBRs and array schema name [#784](https://github.com/TileDB-Inc/TileDB-Py/pull/784)

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
