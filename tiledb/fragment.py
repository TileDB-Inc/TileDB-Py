import pprint
import warnings

import tiledb
from tiledb import _fragment

"""
A high level wrapper around the Pybind11 fragment.cc implementation to ease usability of retrieving information from all fragments for a given array.
"""


class FragmentInfoList:
    """
    Class representing an ordered list of FragmentInfo objects.
    """

    def __init__(self, array_uri, ctx=None):
        if ctx is None:
            ctx = tiledb.default_ctx()

        schema = tiledb.ArraySchema.load(array_uri, ctx=ctx)

        self.array_uri = array_uri

        fi = _fragment.info(self.array_uri, ctx)
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
        self.to_vacuum_num = fi.to_vacuum_num()
        self.to_vacuum_uri = fi.to_vacuum_uri() if self.to_vacuum_num > 0 else []

    @property
    def non_empty_domain(self):
        warnings.warn(
            "FragmentInfoList.non_empty_domain is deprecated; "
            "please use FragmentInfoList.nonempty_domain",
            DeprecationWarning,
        )
        return self.nonempty_domain

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
    Class representing the metadata for a single fragment such as the fragment
    URI, whether it is sparse or dense, timestamp, number of cells, etc.
    """

    def __init__(self, fragments: FragmentInfoList, num):
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
        self.to_vacuum_num = fragments.to_vacuum_num
        self.to_vacuum_uri = (
            fragments.to_vacuum_uri[num] if self.to_vacuum_num > 0 else []
        )

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


def FragmentsInfo(array_uri, ctx=None):
    """
    Deprecated in 0.8.8.

    Renamed to FragmentInfoList to make name more distinguishable from FragmentInfo.
    """

    warnings.warn(
        "FragmentsInfo is deprecated; please use FragmentInfoList",
        DeprecationWarning,
    )

    if ctx is None:
        ctx = tiledb.default_ctx()

    return FragmentInfoList(array_uri, ctx)
