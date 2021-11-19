import pprint
import warnings
import numpy as np

import tiledb
from tiledb.main import PyFragmentInfo

"""
Retrieves information from all fragments for a given array.
"""


class FragmentInfoList:
    """
    Class representing an ordered list of FragmentInfo objects.

    :param array_uri: URI for the TileDB array (any supported TileDB URI)
    :type array_uri: str
    :param include_mbrs: (default False) include minimum bounding rectangles in FragmentInfo result
    :type include_mbrs: bool
    :param ctx: A TileDB context
    :type ctx: tiledb.Ctx

    :ivar uri: URIs of fragments
    :ivar version: Fragment version of each fragment
    :ivar nonempty_domain: Non-empty domain of each fragment
    :ivar cell_num: Number of cells in each fragment
    :ivar timestamp_range: Timestamp range of when each fragment was written
    :ivar sparse: For each fragment, True if fragment is sparse, else False
    :ivar has_consolidated_metadata: For each fragment, True if fragment has consolidated fragment metadata, else False
    :ivar unconsolidated_metadata_num: Number of unconsolidated metadata fragments in each fragment
    :ivar to_vacuum: URIs of already consolidated fragments to vacuum
    :ivar mbrs: (TileDB Embedded 2.5.0+ only) The mimimum bounding rectangle of each fragment; only present when `include_mbrs=True`
    :ivar array_schema_name: (TileDB Embedded 2.5.0+ only) The array schema's name

    **Example:**

    >>> import tiledb, numpy as np, tempfile
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     # The array will be 4x4 with dimensions "rows" and "cols", with domain [1,4] and space tiles 2x2
    ...     dom = tiledb.Domain(
    ...         tiledb.Dim(name="rows", domain=(1, 4), tile=2, dtype=np.int32),
    ...         tiledb.Dim(name="cols", domain=(1, 4), tile=2, dtype=np.int32),
    ...     )
    ...     # The array will be dense with a single attribute "a" so each (i,j) cell can store an integer.
    ...     schema = tiledb.ArraySchema(
    ...         domain=dom, sparse=False, attrs=[tiledb.Attr(name="a", dtype=np.int32)]
    ...     )
    ...     # Set URI of the array
    ...     uri = tmp + "/array"
    ...     # Create the (empty) array on disk.
    ...     tiledb.Array.create(uri, schema)
    ...
    ...     # Write three fragments to the array
    ...     with tiledb.DenseArray(uri, mode="w") as A:
    ...         A[1:3, 1:5] = np.array(([1, 2, 3, 4, 5, 6, 7, 8]))
    ...     with tiledb.DenseArray(uri, mode="w") as A:
    ...         A[2:4, 2:4] = np.array(([101, 102, 103, 104]))
    ...     with tiledb.DenseArray(uri, mode="w") as A:
    ...         A[3:4, 4:5] = np.array(([202]))
    ...
    ...     # tiledb.array_fragments() requires TileDB-Py version > 0.8.5
    ...     fragments_info = tiledb.array_fragments(uri)
    ...
    ...     "====== FRAGMENTS  INFO ======"
    ...     f"number of fragments: {len(fragments_info)}"
    ...     f"nonempty domains: {fragments_info.nonempty_domain}"
    ...     f"sparse fragments: {fragments_info.sparse}"
    ...
    ...     for fragment in fragments_info:
    ...         f"===== FRAGMENT NUMBER {fragment.num} ====="
    ...         f"is sparse: {fragment.sparse}"
    ...         f"cell num: {fragment.cell_num}"
    ...         f"has consolidated metadata: {fragment.has_consolidated_metadata}"
    ...         f"nonempty domain: {fragment.nonempty_domain}"
    '====== FRAGMENTS  INFO ======'
    'number of fragments: 3'
    'nonempty domains: (((1, 2), (1, 4)), ((2, 3), (2, 3)), ((3, 3), (4, 4)))'
    'sparse fragments: (False, False, False)'
    '===== FRAGMENT NUMBER 0 ====='
    'is sparse: False'
    'cell num: 8'
    'has consolidated metadata: False'
    'nonempty domain: ((1, 2), (1, 4))'
    '===== FRAGMENT NUMBER 1 ====='
    'is sparse: False'
    'cell num: 16'
    'has consolidated metadata: False'
    'nonempty domain: ((2, 3), (2, 3))'
    '===== FRAGMENT NUMBER 2 ====='
    'is sparse: False'
    'cell num: 4'
    'has consolidated metadata: False'
    'nonempty domain: ((3, 3), (4, 4))'

    """

    def __init__(self, array_uri, include_mbrs=False, ctx=None):
        if ctx is None:
            ctx = tiledb.default_ctx()

        schema = tiledb.ArraySchema.load(array_uri, ctx=ctx)

        self.array_uri = array_uri

        fi = PyFragmentInfo(self.array_uri, schema, include_mbrs, ctx)

        self.__nums = fi.get_num_fragments()
        self.uri = fi.get_uri()
        self.version = fi.get_version()
        self.nonempty_domain = fi.get_nonempty_domain()
        self.cell_num = fi.get_cell_num()
        self.timestamp_range = fi.get_timestamp_range()
        self.sparse = fi.get_sparse()
        self.unconsolidated_metadata_num = fi.get_unconsolidated_metadata_num()
        self.has_consolidated_metadata = fi.get_has_consolidated_metadata()
        self.to_vacuum = fi.get_to_vacuum()

        if include_mbrs:
            if tiledb.libtiledb.version() >= (2, 5, 0):
                self.mbrs = fi.get_mbrs()
            else:
                warnings.warn(
                    "MBRs for fragments not available; "
                    "please install libtiledb 2.5.0+",
                    UserWarning,
                )

        if tiledb.libtiledb.version() >= (2, 5, 0):
            self.array_schema_name = fi.get_array_schema_name()

    @property
    def non_empty_domain(self):
        warnings.warn(
            "FragmentInfoList.non_empty_domain is deprecated; "
            "please use FragmentInfoList.nonempty_domain",
            DeprecationWarning,
        )
        return self.nonempty_domain

    @property
    def to_vacuum_num(self):
        warnings.warn(
            "FragmentInfoList.to_vacuum_num is deprecated; "
            "please use len(FragmentInfoList.to_vacuum)",
            DeprecationWarning,
        )
        return len(self.to_vacuum)

    @property
    def to_vacuum_uri(self):
        warnings.warn(
            "FragmentInfoList.to_vacuum_uri is deprecated; "
            "please use FragmentInfoList.to_vacuum",
            DeprecationWarning,
        )
        return self.to_vacuum

    @property
    def dense(self):
        warnings.warn(
            "FragmentInfoList.dense is deprecated; "
            "please use FragmentInfoList.sparse",
            DeprecationWarning,
        )
        return list(~np.array(self.sparse))

    def __getattr__(self, name):
        if name == "mbrs":
            raise AttributeError(
                "'FragmentInfoList' object has no attribute 'mbrs'. "
                "(Hint: retrieving minimum bounding rectangles is disabled "
                "by default to optimize speed and space. "
                "Use tiledb.array_fragments(include_mbrs=True) to enable)"
            )
        return self.__getattribute__(name)

    def __iter__(self):
        return FragmentsInfoIterator(self)

    def __getitem__(self, key):
        if isinstance(key, slice):
            # Get the start, stop, and step from the slice
            return [FragmentInfo(self, idx) for idx in range(*key.indices(len(self)))]
        elif isinstance(key, int):
            return FragmentInfo(self, key)
        else:
            raise TypeError("Invalid argument type.")

    def __len__(self):
        return self.__nums

    def __repr__(self):
        public_attrs = {
            key: value
            for (key, value) in self.__dict__.items()
            if not key.startswith("_")
        }
        return pprint.PrettyPrinter().pformat(public_attrs)


class FragmentsInfoIterator:
    """
    Iterator class for the FragmentsInfo container.
    """

    def __init__(self, fragments):
        self._fragments = fragments
        self._index = 0

    def __next__(self):
        if self._index < len(self._fragments):
            fi = FragmentInfo(self._fragments, self._index)
            self._index += 1
            return fi
        raise StopIteration


class FragmentInfo:
    """
    Class representing the metadata for a single fragment. See :py:class:`tiledb.FragmentInfoList` for example of usage.

    :ivar uri: URIs of fragments
    :ivar version: Fragment version of each fragment
    :ivar nonempty_domain: Non-empty domain of each fragment
    :ivar cell_num: Number of cells in each fragment
    :ivar timestamp_range: Timestamp range of when each fragment was written
    :ivar sparse: For each fragment, True if fragment is sparse, else False
    :ivar has_consolidated_metadata: For each fragment, True if fragment has consolidated fragment metadata, else False
    :ivar unconsolidated_metadata_num: Number of unconsolidated metadata fragments in each fragment
    :ivar to_vacuum: URIs of already consolidated fragments to vacuum
    :ivar mbrs: (TileDB Embedded 2.5.0+ only) The mimimum bounding rectangle of each fragment; only present when `include_mbrs=True`
    :ivar array_schema_name: (TileDB Embedded 2.5.0+ only) The array schema's name
    """

    def __init__(self, fragments: FragmentInfoList, num):
        self._frags = fragments
        self.num = num
        self.uri = fragments.uri[num]
        self.version = fragments.version[num]
        self.nonempty_domain = fragments.nonempty_domain[num]
        self.cell_num = fragments.cell_num[num]
        self.timestamp_range = fragments.timestamp_range[num]
        self.sparse = fragments.sparse[num]
        self.has_consolidated_metadata = fragments.has_consolidated_metadata[num]
        self.unconsolidated_metadata_num = fragments.unconsolidated_metadata_num
        self.array_schema_name = fragments.array_schema_name[num]

        if hasattr(fragments, "mbrs"):
            self.mbrs = fragments.mbrs[num]

        if hasattr(fragments, "array_schema_name"):
            self.array_schema_name = fragments.array_schema_name[num]

    def __repr__(self):
        return pprint.PrettyPrinter().pformat(self.__dict__)

    def __getattr__(self, name):
        if name == "mbrs":
            raise AttributeError(
                "'FragmentInfo' object has no attribute 'mbrs'. "
                "(Hint: retrieving minimum bounding rectangles is disabled "
                "by default to optimize speed and space. "
                "Use tiledb.array_fragments(include_mbrs=True) to enable)"
            )
        return self.__getattribute__(name)

    @property
    def non_empty_domain(self):
        warnings.warn(
            "FragmentInfo.non_empty_domain is deprecated; "
            "please use FragmentInfo.nonempty_domain",
            DeprecationWarning,
        )
        return self.nonempty_domain

    @property
    def to_vacuum_num(self):
        warnings.warn(
            "FragmentInfo.to_vacuum_num is deprecated; "
            "please use len(FragmentInfoList.to_vacuum)",
            DeprecationWarning,
        )
        return len(self._frags.to_vacuum)

    @property
    def to_vacuum_uri(self):
        warnings.warn(
            "FragmentInfo.to_vacuum_uri is deprecated; "
            "please use FragmentInfoList.to_vacuum",
            DeprecationWarning,
        )
        return self._frags.to_vacuum

    @property
    def to_vacuum_uri(self):
        warnings.warn(
            "FragmentInfo.dense is deprecated; please use FragmentInfo.sparse",
            DeprecationWarning,
        )
        return not self._frags.sparse


def FragmentsInfo(array_uri, ctx=None):
    """
    Deprecated in 0.8.8.

    Renamed to FragmentInfoList to make name more distinguishable from FragmentInfo.
    """

    warnings.warn(
        "FragmentsInfo is deprecated; please use FragmentInfoList", DeprecationWarning
    )

    if ctx is None:
        ctx = tiledb.default_ctx()

    return FragmentInfoList(array_uri, ctx)
