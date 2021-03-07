import tiledb
from tiledb import _fragment

import pprint

"""
A high level wrapper around the Pybind11 fragment.cc implementation to ease usability of retrieving information from all fragments for a given array.
"""


class FragmentsInfo:
    """
    Class representing a container of fragment info objects.
    """

    def __init__(self, array_uri, ctx=None):
        schema = tiledb.ArraySchema.load(array_uri)

        self.array_uri = array_uri

        fi = _fragment.info(self.array_uri, ctx)
        fi.load()

        self.uri = fi.fragment_uri()
        self.__nums = fi.fragment_num()
        self.version = fi.version()
        self.non_empty_domain = fi.get_non_empty_domain(schema)
        self.cell_num = fi.cell_num()
        self.timestamp_range = fi.timestamp_range()
        self.dense = fi.dense()
        self.sparse = fi.sparse()
        self.has_consolidated_metadata = fi.has_consolidated_metadata()
        self.unconsolidated_metadata_num = fi.unconsolidated_metadata_num()
        self.to_vacuum_num = fi.to_vacuum_num()
        self.to_vacuum_uri = fi.to_vacuum_uri() if self.to_vacuum_num > 0 else []

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
    Interator class for the FragmentsInfo container.
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
    Class representing a single fragment info object.
    """

    def __init__(self, fragments: FragmentsInfo, num):
        self.num = num
        self.uri = fragments.uri[num]
        self.version = fragments.version[num]
        self.non_empty_domain = fragments.non_empty_domain[num]
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
