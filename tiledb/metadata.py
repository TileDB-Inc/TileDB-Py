from __future__ import absolute_import

import sys
import array as cparray
from tiledb import libtiledb as libmetadata
from tiledb.libtiledb import ustring

class Metadata(object):

    def __init__(self, array):
        self.array = array

    def __setitem__(self, key, value):
        """
        Implementation of [key] <- val (dict item assignment)

        :param key: key to set
        :param value: corresponding value
        :return: None
        """

        libmetadata.put_metadata(self.array, key, value)

    def __getitem__(self, key):
        """
        Implementation of [key] -> val (dict item retrieval)
        :param key:
        :return:
        """
        if not (isinstance(key, str) or isinstance(key, unicode)):
            raise ValueError("Unexpected key type '{}': expected str "
                             "type".format(type(key)))

        # `get_metadata` expects unicode
        key = ustring(key)
        v = libmetadata.get_metadata(self.array, key)

        if v is None:
            raise TileDBError("Failed to unpack value for key: '{}'".format(key))

        return v

    def __contains__(self, key):
        """
        Returns True if 'key' is found in metadata store.
        Provides support for python 'in' syntax ('k in A.meta')

        :param key: Target key to check.
        :return:
        """

        try:
            self[key]
        except KeyError:
            return False

        return True

    def consolidate(self):
        """
        Consolidate array metadata. Array must be closed.

        :return:
        """

        # TODO: ensure that the array is not x-locked?

        libmetadata.consolidate_metadata(self.array)

    def __delitem__(self, key):
        """
        Remove key from metadata.

        **Example:**

        >>> # given A = tiledb.open(uri, ...)
        >>> del A.meta['key']

        :param key:
        :return:
        """

        libmetadata.del_metadata(self.array, key)

    def __len__(self):
        """
        :return: length of metadata store
        """

        return libmetadata.len_metadata(self.array)

    def keys(self):
        """
        Return metadata keys as list.

        :return: List of array metadata keys.
        """

        return libmetadata.load_metadata(self.array, unpack=False)

    def values(self):
        # TODO this should be an iterator

        data = libmetadata.load_metadata(self.array, unpack=True)
        return data.values()

    def pop(self, key, default=None):
        raise NotImplementedError("dict.pop requires read-write access to array")

    def items(self):
        # TODO this should be an iterator
        data = libmetadata.load_metadata(self.array, unpack=True)
        return tuple( (k, data[k]) for k in data.keys() )

    def _set_numpy(self, key, value):
        """
        Test helper: directly set meta key-value from a NumPy array.
        Key type and array dimensionality are checked, but no other type-checking
        is done.

        :param key: key
        :param arr: 1d NumPy ndarray
        :return:
        """

        libmetadata._set_metadata_numpy(self.array, key, value)