from typing import MutableMapping, Optional, Union

import numpy as np

import tiledb.cc as lt

from .ctx import Config, Ctx, CtxMixin, default_ctx
from .datatypes import DataType
from .object import Object


class Group(CtxMixin, lt.Group):
    """
    Support for organizing multiple arrays in arbitrary directory hierarchies.

    Group members may be any number of nested groups and arrays. Members are stored as tiledb.Objects which indicate the member's URI and type.

    Groups may contain associated metadata similar to array metadata where
    keys are strings. Singleton values may be of type int, float, str, or bytes. Multiple values of the same type may be placed in containers of type list, tuple, or 1-D np.ndarray. The values within containers are limited to type int or float.

    See more at: https://docs.tiledb.com/main/background/key-concepts-and-data-format#arrays-and-groups

    :param uri: The URI to the Group
    :type uri: str
    :param mode: Read mode ('r'), write mode ('w'), or modify exclusive ('m')
    :type mode: str
    :param config: A TileDB config
    :type config: Config or dict
    :param ctx: A TileDB context
    :type ctx: tiledb.Ctx

    **Example:**

    >>> # Create a group
    >>> grp_path = "root_group"
    >>> tiledb.Group.create(grp_path)
    >>> grp = tiledb.Group(grp_path, "w")
    >>>
    >>> # Create an array and add as a member to the group
    >>> array_path = "array.tdb"
    >>> domain = tiledb.Domain(tiledb.Dim(domain=(1, 8), tile=2))
    >>> a1 = tiledb.Attr("val", dtype="f8")
    >>> schema = tiledb.ArraySchema(domain=domain, attrs=(a1,))
    >>> tiledb.Array.create(array_path, schema)
    >>> grp.add(array_path)
    >>>
    >>> # Create a group and add as a subgroup
    >>> subgrp_path = "sub_group"
    >>> tiledb.Group.create(subgrp_path)
    >>> grp.add(subgrp_path)
    >>>
    >>> # Add metadata to the subgroup
    >>> grp.meta["ints"] = [1, 2, 3]
    >>> grp.meta["str"] = "string_metadata"
    >>> grp.close()
    >>>
    >>> grp.open("r")
    >>> # Dump all the members in string format
    >>> mbrs_repr = grp
    >>> # Or create a list of Objects in the Group
    >>> mbrs_iter = list(grp)
    >>> # Get the first member's uri and type
    >>> member_uri, member_type = grp[0].uri, grp[0].type
    >>> grp.close()
    >>>
    >>> # Remove the subgroup
    >>> grp.open("w")
    >>> grp.remove(subgrp_path)
    >>> grp.close()
    >>>
    >>> # Delete the subgroup
    >>> grp.open("m")
    >>> grp.delete(subgrp_path)
    >>> grp.close()
    """

    _NP_DATA_PREFIX = "__np_flat_"
    _NP_SHAPE_PREFIX = "__np_shape_"

    _mode_to_query_type = {
        "r": lt.QueryType.READ,
        "w": lt.QueryType.WRITE,
        "m": lt.QueryType.MODIFY_EXCLUSIVE,
    }

    _query_type_to_mode = {t: m for m, t in _mode_to_query_type.items()}

    __was_deleted__ = False

    class GroupMetadata(MutableMapping):
        """
        Holds metadata for the associated Group in a dictionary-like structure.
        """

        GroupMetadataValueType = Union[int, float, str, bytes, np.ndarray]

        def __init__(self, group: "Group"):
            self._group = group

        def __setitem__(self, key: str, value: GroupMetadataValueType):
            """
            :param str key: Key for the Group metadata entry
            :param value: Value for the Group metadata entry
            :type value: Union[int, float, str, bytes, np.ndarray]

            """
            if not isinstance(key, str):
                raise TypeError(f"Unexpected key type '{type(key)}': expected str")

            # ensure previous key(s) are deleted (e.g. in case of replacing a
            # non-numpy value with a numpy value or vice versa)
            del self[key]

            put_metadata = self._group._put_metadata
            if isinstance(value, np.ndarray):
                flat_value = value.ravel()
                put_metadata(f"{Group._NP_DATA_PREFIX}{key}", flat_value)
                if value.shape != flat_value.shape:
                    # If the value is not a 1D ndarray, store its associated shape.
                    # The value's shape will be stored as separate metadata with the correct prefix.
                    self.__setitem__(f"{Group._NP_SHAPE_PREFIX}{key}", value.shape)
            else:
                from .metadata import pack_metadata_val

                packed_buf = pack_metadata_val(value)
                tiledb_type = packed_buf.tdbtype
                value_num = packed_buf.value_num
                data_view = packed_buf.data

                put_metadata(key, tiledb_type, value_num, data_view)

        def __getitem__(self, key: str, include_type=False) -> GroupMetadataValueType:
            """
            :param str key: Key of the Group metadata entry
            :rtype: Union[int, float, str, bytes, np.ndarray]
            :return: The value associated with the key

            """
            if not isinstance(key, str):
                raise TypeError(f"Unexpected key type '{type(key)}': expected str")

            if self._group._has_metadata(key):
                data, tdb_type = self._group._get_metadata(key, False)
            elif self._group._has_metadata(f"{Group._NP_DATA_PREFIX}{key}"):
                data, tdb_type = self._group._get_metadata(
                    f"{Group._NP_DATA_PREFIX}{key}", True
                )
                # reshape numpy array back to original shape, if needed
                shape_key = f"{Group._NP_SHAPE_PREFIX}{key}"
                if self._group._has_metadata(shape_key):
                    shape, tdb_type = self._group._get_metadata(shape_key, False)
                    data = data.reshape(shape)
            else:
                raise KeyError(f"KeyError: {key}")

            return (data, tdb_type) if include_type else data

        def __delitem__(self, key: str):
            """Removes the entry from the Group metadata.

            :param str key: Key of the Group metadata entry

            """
            if not isinstance(key, str):
                raise TypeError(f"Unexpected key type '{type(key)}': expected str")

            # key may be stored as is or it may be prefixed (for numpy values)
            # we don't know this here so delete all potential internal keys
            for k in key, Group._NP_DATA_PREFIX + key, Group._NP_SHAPE_PREFIX + key:
                self._group._delete_metadata(k)

        def __contains__(self, key: str) -> bool:
            """
            :param str key: Key of the Group metadata entry
            :rtype: bool
            :return: True if the key is in the Group metadata, otherwise False

            """
            if not isinstance(key, str):
                raise TypeError(f"Unexpected key type '{type(key)}': expected str")

            # key may be stored as is or it may be prefixed (for numpy values)
            # we don't know this here so check all potential internal keys
            return self._group._has_metadata(key) or self._group._has_metadata(
                f"{Group._NP_DATA_PREFIX}{key}"
            )

        def __len__(self) -> int:
            """
            :rtype: int
            :return: Number of entries in the Group metadata

            """
            num = self._group._metadata_num()
            # subtract the _NP_SHAPE_PREFIX prefixed keys
            for key in self._iter(keys_only=True):
                if key.startswith(Group._NP_SHAPE_PREFIX):
                    num -= 1

            return num

        def _iter(self, keys_only: bool = True, dump: bool = False):
            """
            Iterate over Group metadata keys or (key, value) tuples
            :param keys_only: whether to yield just keys or values too
            :param dump: whether to yield a formatted string for each metadata entry
            """
            if keys_only and dump:
                raise ValueError("keys_only and dump cannot both be True")

            metadata_num = self._group._metadata_num()
            for i in range(metadata_num):
                key = self._group._get_key_from_index(i)

                if keys_only:
                    yield key
                else:
                    val, val_dtype = self.__getitem__(key, include_type=True)

                    if dump:
                        yield (
                            "### Array Metadata ###\n"
                            f"- Key: {key}\n"
                            f"- Value: {val}\n"
                            f"- Type: {val_dtype}\n"
                        )
                    else:
                        yield key, val

        def __iter__(self):
            np_data_prefix_len = len(Group._NP_DATA_PREFIX)
            for key in self._iter(keys_only=True):
                if key.startswith(Group._NP_DATA_PREFIX):
                    yield key[np_data_prefix_len:]
                elif not key.startswith(Group._NP_SHAPE_PREFIX):
                    yield key
                # else: ignore the shape keys

        def __repr__(self):
            return str(dict(self))

        def setdefault(self, key, default=None):
            raise NotImplementedError(
                "Group.GroupMetadata.setdefault requires read-write access"
            )

        def pop(self, key, default=None):
            raise NotImplementedError(
                "Group.GroupMetadata.pop requires read-write access"
            )

        def popitem(self):
            raise NotImplementedError(
                "Group.GroupMetadata.popitem requires read-write access"
            )

        def clear(self):
            raise NotImplementedError(
                "Group.GroupMetadata.clear requires read-write access"
            )

        def dump(self):
            """Output information about all group metadata to stdout."""
            for metadata in self._iter(keys_only=False, dump=True):
                print(metadata)

    def __init__(
        self,
        uri: str,
        mode: str = "r",
        config: Config = None,
        ctx: Optional[Ctx] = None,
    ):
        if mode not in Group._mode_to_query_type:
            raise ValueError(f"invalid mode {mode}")
        query_type = Group._mode_to_query_type[mode]

        if config is None:
            super().__init__(ctx, uri, query_type)
        else:
            super().__init__(ctx, uri, query_type, config)

        self._meta = self.GroupMetadata(self)

    @staticmethod
    def create(uri: str, ctx: Optional[Ctx] = None):
        """
        Create a new Group.

        :param uri: The URI to the to-be created Group
        :type uri: str
        :param ctx: A TileDB context
        :type ctx: tiledb.Ctx
        """
        ctx = ctx or default_ctx()
        lt.Group._create(ctx, uri)

    def open(self, mode: str = "r"):
        """
        Open a Group in read mode ("r") or write mode ("w").

        :param mode: Read mode ('r') or write mode ('w')
        :type mode: str
        """
        if mode not in Group._mode_to_query_type:
            raise ValueError(f"invalid mode {mode}")
        query_type = Group._mode_to_query_type[mode]

        self._open(query_type)

    def close(self):
        """
        Close a Group.
        """
        self._close()

    def add(self, uri: str, name: str = None, relative: bool = False):
        """
        Adds a member to the Group.

        :param uri: The URI of the member to add
        :type uri: str
        :param relative: Whether the path of the URI is a relative path (default=relative: False)
        :type relative: bool
        :param name: An optional name for the Group (default=None)
        :type name: str
        """
        if name:
            self._add(uri, relative, name)
        else:
            self._add(uri, relative)

    def delete(self, recursive: bool = False):
        """
        Delete a Group. The group needs to be opened in 'm' mode.

        :param uri: The URI of the group to delete
        """
        self._delete_group(self.uri, recursive)
        self.__was_deleted__ = True

    def __getitem__(self, member: Union[int, str]) -> Object:
        """
        Retrieve a member from the Group as an Object.

        :param member: The index or name of the member
        :type member: Union[int, str]
        :return: The member as an Object
        :rtype: Object
        """
        if not isinstance(member, (int, str)):
            raise TypeError(
                f"Unexpected member type '{type(member)}': expected int or str"
            )

        obj = self._member(member)
        return Object(obj._type, obj._uri, obj._name)

    def __len__(self) -> int:
        """
        :rtype: int
        :return: Number of members in the Group

        """
        return self._member_count()

    def remove(self, member: str):
        """
        Remove a member from the Group.

        :param member: The URI or name of the member
        :type member: str
        """
        if not isinstance(member, str):
            raise TypeError(f"Unexpected member type '{type(member)}': expected str")

        self._remove(member)

    def __delitem__(self, uri: str):
        """
        Remove a member from the group.

        :param uri: The URI to the member
        :type uri: str
        """
        self._remove(uri)

    def __iter__(self):
        return iter(self[i] for i in range(len(self)))

    def __contains__(self, member: str) -> bool:
        """
        :return: Whether the Group contains a member with the given name
        :rtype: bool
        """
        return self._has_member(member)

    def __repr__(self):
        # use safe repr if pybind11 constructor failed
        if self._ctx is None:
            return object.__repr__(self)

        return self._dump(True)

    def __enter__(self):
        """
        The `__enter__` and `__exit__` methods allow TileDB groups to be opened (and auto-closed)
        using with-as syntax.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        The `__enter__` and `__exit__` methods allow TileDB groups to be opened (and auto-closed)
        using with-as syntax.
        """
        # Don't close if this was a delete operation: the group will be closed
        # automatically.
        if not (hasattr(self, "__deleted") or self.__was_deleted__):
            self.__was_deleted__ = False
            self.close()

    @property
    def meta(self) -> GroupMetadata:
        """
        :return: The Group's metadata as a key-value structure
        :rtype: GroupMetadata
        """
        return self._meta

    @property
    def isopen(self) -> bool:
        """
        :return: Whether or not the Group is open
        :rtype: bool
        """
        return self._isopen

    @property
    def uri(self) -> str:
        """
        :return: URI of the Group
        :rtype: str
        """
        return self._uri

    @property
    def mode(self) -> str:
        """
        :return: Read mode ('r'), write mode ('w'), or modify exclusive ('m')
        :rtype: str
        """
        return self._query_type_to_mode[self._query_type]

    def is_relative(self, name: str) -> bool:
        """
        :param name: Name of member to retrieve associated relative indicator
        :type name: str
        :return: Whether the attribute is relative
        :rtype: bool
        """
        return self._is_relative(name)

    def set_config(self, cfg: Config):
        """
        :param cfg: Config to set on the Group
        :type cfg: Config
        """
        if self.isopen:
            raise ValueError(
                "`set_config` can only be used on closed groups. "
                "Use `group.cl0se()` or Group(.., closed=True)"
            )
        self._set_config(cfg)

    @staticmethod
    def consolidate_metadata(
        uri: str, config: Config = None, ctx: Optional[Ctx] = None
    ):
        """
        Consolidate the group metadata.

        :param uri: The URI of the TileDB group to be consolidated
        :type uri: str
        :param config: Optional configuration parameters for the consolidation
        :type config: Config
        :param ctx: Optional TileDB context
        :type ctx: Ctx
        """
        if ctx is None:
            ctx = default_ctx()

        lt.Group._consolidate_metadata(ctx, uri, config)

    @staticmethod
    def vacuum_metadata(uri: str, config: Config = None, ctx: Optional[Ctx] = None):
        """
        Vacuum the group metadata.

        :param uri: The URI of the TileDB group to be vacuum
        :type uri: str
        :param config: Optional configuration parameters for the vacuuming
        :type config: Config
        :param ctx: Optional TileDB context
        :type ctx: Ctx
        """
        if ctx is None:
            ctx = default_ctx()

        lt.Group._vacuum_metadata(ctx, uri, config)
