from typing import MutableMapping, Optional, Union

import numpy as np

import tiledb
import tiledb.libtiledb as lt

from .ctx import Config, Ctx, default_ctx
from .datatypes import DataType


class PackedBuffer:
    def __init__(self, data, tdbtype, value_num):
        self.data = data
        self.tdbtype = tdbtype
        self.value_num = value_num


def pack_metadata_val(value) -> PackedBuffer:
    if isinstance(value, bytes):
        return PackedBuffer(value, lt.DataType.BLOB, len(value))

    if isinstance(value, str):
        value = value.encode("UTF-8")
        return PackedBuffer(value, lt.DataType.STRING_UTF8, len(value))

    if not isinstance(value, (list, tuple)):
        value = (value,)

    if not value:
        # special case for empty values
        return PackedBuffer(b"", lt.DataType.INT32, 0)

    val0 = value[0]
    if not isinstance(val0, (int, float)):
        raise TypeError(f"Unsupported item type '{type(val0)}'")

    tiledb_type = lt.DataType.INT64 if isinstance(val0, int) else lt.DataType.FLOAT64
    python_type = int if isinstance(val0, int) else float
    numpy_dtype = np.int64 if isinstance(val0, int) else np.float64
    data_view = memoryview(
        bytearray(len(value) * tiledb.main.datatype_size(tiledb_type))
    )

    for i, val in enumerate(value):
        if not isinstance(val, python_type):
            raise TypeError(f"Mixed-type sequences are not supported: {value}")
        data_view[i * 8 : (i + 1) * 8] = numpy_dtype(val).tobytes()

    return PackedBuffer(bytes(data_view), tiledb_type, len(value))


class Metadata(MutableMapping):
    """
    Holds metadata for the associated Array or Group in a dictionary-like structure.
    """

    _NP_DATA_PREFIX = "__np_flat_"
    _NP_SHAPE_PREFIX = "__np_shape_"

    MetadataValueType = Union[int, float, str, bytes, np.ndarray]

    def __init__(self, array_or_group):
        self._array_or_group = array_or_group

    def __setitem__(self, key: str, value: MetadataValueType):
        """
        :param str key: Key for the metadata entry
        :param value: Value for the metadata entry
        :type value: Union[int, float, str, bytes, np.ndarray]

        """
        if not isinstance(key, str):
            raise TypeError(f"Unexpected key type '{type(key)}': expected str")

        # ensure previous key(s) are deleted (e.g. in case of replacing a
        # non-numpy value with a numpy value or vice versa)
        del self[key]

        put_metadata = self._array_or_group._put_metadata
        if isinstance(value, np.ndarray):
            flat_value = value.ravel()
            put_metadata(f"{self._NP_DATA_PREFIX}{key}", flat_value)
            if value.shape != flat_value.shape:
                # If the value is not a 1D ndarray, store its associated shape.
                # The value's shape will be stored as separate metadata with the correct prefix.
                self.__setitem__(f"{self._NP_SHAPE_PREFIX}{key}", value.shape)
        elif isinstance(value, np.generic):
            tiledb_type = DataType.from_numpy(value.dtype).tiledb_type
            if tiledb_type in (lt.DataType.BLOB, lt.DataType.CHAR):
                put_metadata(key, tiledb_type, len(value), value)
            elif tiledb_type == lt.DataType.STRING_UTF8:
                put_metadata(
                    key, lt.DataType.STRING_UTF8, len(value), value.encode("UTF-8")
                )
            else:
                put_metadata(key, tiledb_type, 1, value)
        else:
            packed_buf = pack_metadata_val(value)
            tiledb_type = packed_buf.tdbtype
            value_num = packed_buf.value_num
            data_view = packed_buf.data

            put_metadata(key, tiledb_type, value_num, data_view)

    def __getitem__(self, key: str, include_type=False) -> MetadataValueType:
        """
        :param str key: Key of the metadata entry
        :rtype: Union[int, float, str, bytes, np.ndarray]
        :return: The value associated with the key

        """
        if not isinstance(key, str):
            raise TypeError(f"Unexpected key type '{type(key)}': expected str")

        if self._array_or_group._has_metadata(key):
            data, tdb_type = self._array_or_group._get_metadata(key, False)
            dtype = DataType.from_tiledb(tdb_type).np_dtype
            # we return all int and float values as numpy scalars
            if dtype.kind in ("i", "f") and not isinstance(data, tuple):
                data = np.dtype(dtype).type(data)
        elif self._array_or_group._has_metadata(f"{self._NP_DATA_PREFIX}{key}"):
            data, tdb_type = self._array_or_group._get_metadata(
                f"{self._NP_DATA_PREFIX}{key}", True
            )
            # reshape numpy array back to original shape, if needed
            shape_key = f"{self._NP_SHAPE_PREFIX}{key}"
            if self._array_or_group._has_metadata(shape_key):
                shape, tdb_type = self._array_or_group._get_metadata(shape_key, False)
                data = data.reshape(shape)
        else:
            raise KeyError(f"KeyError: {key}")

        return (data, tdb_type) if include_type else data

    def __delitem__(self, key: str):
        """Removes the entry from the metadata.

        :param str key: Key of the metadata entry

        """
        if not isinstance(key, str):
            raise TypeError(f"Unexpected key type '{type(key)}': expected str")

        # key may be stored as is or it may be prefixed (for numpy values)
        # we don't know this here so delete all potential internal keys
        for k in key, self._NP_DATA_PREFIX + key, self._NP_SHAPE_PREFIX + key:
            self._array_or_group._delete_metadata(k)

    def __contains__(self, key: str) -> bool:
        """
        :param str key: Key of the metadata entry
        :rtype: bool
        :return: True if the key is in the metadata, otherwise False

        """
        if not isinstance(key, str):
            raise TypeError(f"Unexpected key type '{type(key)}': expected str")

        # key may be stored as is or it may be prefixed (for numpy values)
        # we don't know this here so check all potential internal keys
        return self._array_or_group._has_metadata(
            key
        ) or self._array_or_group._has_metadata(f"{self._NP_DATA_PREFIX}{key}")

    def __len__(self) -> int:
        """
        :rtype: int
        :return: Number of entries in the metadata

        """
        num = self._array_or_group._metadata_num()
        # subtract the _NP_SHAPE_PREFIX prefixed keys
        for key in self._iter(keys_only=True):
            if key.startswith(self._NP_SHAPE_PREFIX):
                num -= 1

        return num

    def _iter(self, keys_only: bool = True, dump: bool = False):
        """
        Iterate over metadata keys or (key, value) tuples
        :param keys_only: whether to yield just keys or values too
        :param dump: whether to yield a formatted string for each metadata entry
        """
        if keys_only and dump:
            raise ValueError("keys_only and dump cannot both be True")

        metadata_num = self._array_or_group._metadata_num()
        for i in range(metadata_num):
            key = self._array_or_group._get_key_from_index(i)

            if keys_only:
                yield key
            else:
                val, val_dtype = self.__getitem__(key, include_type=True)

                if dump:
                    class_str = (
                        "Array"
                        if isinstance(self._array_or_group, lt.Array)
                        else "Group"
                    )
                    yield (
                        f"### {class_str} Metadata ###\n"
                        f"- Key: {key}\n"
                        f"- Value: {val}\n"
                        f"- Type: {val_dtype}\n"
                    )
                else:
                    yield key, val

    def __iter__(self):
        np_data_prefix_len = len(self._NP_DATA_PREFIX)
        for key in self._iter(keys_only=True):
            if key.startswith(self._NP_DATA_PREFIX):
                yield key[np_data_prefix_len:]
            elif not key.startswith(self._NP_SHAPE_PREFIX):
                yield key
            # else: ignore the shape keys

    def __repr__(self):
        return str(dict(self))

    def consolidate(self, config: Config = None, ctx: Optional[Ctx] = None):
        """
        Consolidate the metadata.

        :param uri: The URI of the TileDB Array or Group to be consolidated
        :type uri: str
        :param config: Optional configuration parameters for the consolidation
        :type config: Config
        :param ctx: Optional TileDB context
        :type ctx: Ctx
        """
        if ctx is None:
            ctx = default_ctx()

        uri = self._array_or_group._uri()

        if isinstance(self._array_or_group, lt.Array):
            lt.Array._consolidate_metadata(ctx, uri, config)
        elif isinstance(self._array_or_group, lt.Group):
            lt.Group._consolidate_metadata(ctx, uri, config)
        else:
            raise ValueError("Unexpected object type")

    def setdefault(self, key, default=None):
        raise NotImplementedError("Metadata.setdefault requires read-write access")

    def pop(self, key, default=None):
        raise NotImplementedError("Metadata.pop requires read-write access")

    def popitem(self):
        raise NotImplementedError("Metadata.popitem requires read-write access")

    def clear(self):
        raise NotImplementedError("Metadata.clear requires read-write access")

    def dump(self):
        """Output information about all metadata to stdout."""
        for metadata in self._iter(keys_only=False, dump=True):
            print(metadata)
