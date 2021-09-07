import pprint
import warnings

import tiledb
from tiledb.main import PyFragmentInfo

"""
Retrieves information from all fragments for a given array.
"""


class FragmentInfoList:
    """
    Class representing an ordered list of FragmentInfo objects.

    :param uri: URIs of fragments
    :param version: Fragment version of each fragment
    :param nonempty_domain: Non-empty domain of each fragment
    :param cell_num: Number of cells in each fragment
    :param timestamp_range: Timestamp range of when each fragment was written
    :param dense: For each fragment, True if fragment is dense, else False
    :param sparse: For each fragment, True if fragment is sparse, else False
    :param has_consolidated_metadata: For each fragment, True if fragment has consolidated fragment metadata, else False
    :param unconsolidated_metadata_num: Number of unconsolidated metadata fragments in each fragment
    :param to_vacuum: URIs of already consolidated fragments to vacuum

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
    ...         f"is dense: {fragment.dense}"
    ...         f"cell num: {fragment.cell_num}"
    ...         f"has consolidated metadata: {fragment.has_consolidated_metadata}"
    ...         f"nonempty domain: {fragment.nonempty_domain}"
    '====== FRAGMENTS  INFO ======'
    'number of fragments: 3'
    'nonempty domains: (((1, 2), (1, 4)), ((2, 3), (2, 3)), ((3, 3), (4, 4)))'
    'sparse fragments: (False, False, False)'
    '===== FRAGMENT NUMBER 0 ====='
    'is dense: True'
    'cell num: 8'
    'has consolidated metadata: False'
    'nonempty domain: ((1, 2), (1, 4))'
    '===== FRAGMENT NUMBER 1 ====='
    'is dense: True'
    'cell num: 16'
    'has consolidated metadata: False'
    'nonempty domain: ((2, 3), (2, 3))'
    '===== FRAGMENT NUMBER 2 ====='
    'is dense: True'
    'cell num: 4'
    'has consolidated metadata: False'
    'nonempty domain: ((3, 3), (4, 4))'

    """

    def __init__(self, array_uri, ctx=None):
        if ctx is None:
            ctx = tiledb.default_ctx()

        schema = tiledb.ArraySchema.load(array_uri, ctx=ctx)

        self.array_uri = array_uri

        fi = PyFragmentInfo(self.array_uri, ctx)
        fi.load()

        self.uri = fi.fragment_uri()
        self.__nums = fi.fragment_num()
        self.version = fi.version()
        self.nonempty_domain = fi.get_non_empty_domain(schema)
        self.cell_num = fi.cell_num()
        self.timestamp_range = fi.timestamp_range()
        self.dense = fi.dense()
        self.sparse = fi.sparse()
        self.has_consolidated_metadata = fi.has_consolidated_metadata()
        self.unconsolidated_metadata_num = fi.unconsolidated_metadata_num()
        self.to_vacuum = fi.to_vacuum_uri()

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

    :param str uri: URIs of fragments
    :param int version: Fragment version of each fragment
    :param nonempty_domain: Non-empty domain of each fragment
    :type nonempty_domain: tuple(numpy scalar, numpy scalar)
    :param cell_num int: Number of cells in each fragment
    :param timestamp_range: Timestamp range of when each fragment was written
    :type timestamp_range: tuple(int, int)
    :param bool dense: True if fragment is dense, else False
    :param bool sparse: True if fragment is sparse, else False
    :param bool has_consolidated_metadata: True if fragment has consolidated metadata, else False
    :param int unconsolidated_metadata_num: Number of unconsolidated metadata fragments
    """

    def __init__(self, fragments: FragmentInfoList, num):
        self._frags = fragments
        self.num = num
        self.uri = fragments.uri[num]
        self.version = fragments.version[num]
        self.nonempty_domain = fragments.nonempty_domain[num]
        self.cell_num = fragments.cell_num[num]
        self.timestamp_range = fragments.timestamp_range[num]
        self.dense = fragments.dense[num]
        self.sparse = fragments.sparse[num]
        self.has_consolidated_metadata = fragments.has_consolidated_metadata[num]
        self.unconsolidated_metadata_num = fragments.unconsolidated_metadata_num

    def __repr__(self):
        return pprint.PrettyPrinter().pformat(self.__dict__)

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
