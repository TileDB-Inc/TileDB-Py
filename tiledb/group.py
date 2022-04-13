from typing import Union, TYPE_CHECKING
import numpy as np

import tiledb.cc as lt
from .ctx import default_ctx

if TYPE_CHECKING:
    from .object import Object


class Group(lt.Group):
    """
    Support for organizing multiple arrays in arbitrary directory hierarchies.
    Group members may be any number of nested groups and arrays. Members are
    stored as `tiledb.Object`s which indicate the member's URI and type.

    Groups may contain associated metadata similar to array metadata where
    keys are strings. Singleton values may be of type int, float, str, or bytes.
    Multiple values of the same type may be placed in containers of type list,
    tuple, or 1-D np.ndarray. The values within containers are limited to type
    int or float.

    See more at:
    https://docs.tiledb.com/main/background/key-concepts-and-data-format#arrays-and-groups

    :param uri: The URI to the group
    :type str
    :param mode: Read mode ('r') or write mode ('w')
    :type str
    :param ctx: A TileDB context
    :type tiledb.Ctx

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
    >>> # Get the first member's uri and type
    >>> grp.open("r")
    >>> member_uri, member_type = grp[0].uri, grp[0].type
    >>> grp.close()
    >>>
    >>> # Remove the subgroup
    >>> grp.open("w")
    >>> grp.remove(subgrp_path)
    >>> grp.close()
    """

    _NP_DATA_PREFIX = "__np_flat_"

    _mode_to_query_type = {
        "r": lt.QueryType.READ,
        "w": lt.QueryType.WRITE,
    }

    _query_type_to_mode = {
        lt.QueryType.READ: "r",
        lt.QueryType.WRITE: "w",
    }

    class GroupMetadata:
        """
        Holds metadata for the associated Group in a dictionary-like structure.
        """

        GroupMetadataValueType = Union[int, float, str, bytes, np.ndarray]

        def __init__(self, group: "Group"):
            self._group = group

        def __setitem__(self, key: str, value: GroupMetadataValueType):
            if not isinstance(key, str):
                raise TypeError(f"Unexpected key type '{type(key)}': expected str")

            # ensure previous key(s) are deleted (e.g. in case of replacing a
            # non-numpy value with a numpy value or vice versa)
            del self[key]

            if isinstance(value, np.ndarray):
                prefix = Group._NP_DATA_PREFIX
                _value = value
            else:
                prefix = ""
                if isinstance(value, (list, tuple)):
                    _value = np.array(value)
                else:
                    _value = np.array([value])

            self._group._put_metadata(f"{prefix}{key}", _value)

        def __getitem__(self, key: str) -> GroupMetadataValueType:
            if not isinstance(key, str):
                raise TypeError(f"Unexpected key type '{type(key)}': expected str")

            if self._group._has_metadata(key):
                data = self._group._get_metadata(key)
                if data.dtype.kind == "U":
                    data = "".join(data)
                elif data.dtype.kind == "S":
                    data = data.tobytes()
                else:
                    data = tuple(data)
                return data[0] if len(data) == 1 else data
            elif self._group._has_metadata(f"{Group._NP_DATA_PREFIX}{key}"):
                return self._group._get_metadata(f"{Group._NP_DATA_PREFIX}{key}")
            else:
                raise KeyError(f"KeyError: {key}")

        def __delitem__(self, key: str):
            if not isinstance(key, str):
                raise TypeError(f"Unexpected key type '{type(key)}': expected str")

            # key may be stored as is or it may be prefixed (for numpy values)
            # we don't know this here so delete all potential internal keys
            self._group._delete_metadata(key)
            self._group._delete_metadata(f"{Group._NP_DATA_PREFIX}{key}")

        def __contains__(self, key: str):
            if not isinstance(key, str):
                raise TypeError(f"Unexpected key type '{type(key)}': expected str")

            # key may be stored as is or it may be prefixed (for numpy values)
            # we don't know this here so check all potential internal keys
            return self._group._has_metadata(key) or self._group._has_metadata(
                f"{Group._NP_DATA_PREFIX}{key}"
            )

        def __len__(self):
            return self._group._metadata_num()

    def __init__(self, uri: str, mode: str = "r", ctx: "Ctx" = None):
        self._ctx = ctx or default_ctx()
        cctx = lt.Context(self._ctx.__capsule__(), False)

        if mode not in Group._mode_to_query_type:
            raise ValueError(f"invalid mode {mode}")
        query_type = Group._mode_to_query_type[mode]

        super().__init__(cctx, uri, query_type)

        self._meta = self.GroupMetadata(self)

    @staticmethod
    def create(uri: str, ctx: "Ctx" = None):
        _ctx = ctx or default_ctx()
        cctx = lt.Context(_ctx.__capsule__(), False)
        lt.Group._create(cctx, uri)

    def open(self, mode: str = "r"):
        if mode not in Group._mode_to_query_type:
            raise ValueError(f"invalid mode {mode}")
        query_type = Group._mode_to_query_type[mode]

        self._open(query_type)

    def __getitem__(self, idx: int) -> "Object":
        from .object import Object

        if not isinstance(idx, int):
            raise TypeError(f"Unexpected idx type '{type(idx)}': expected int")

        obj = self._member(idx)
        return Object(obj._type, obj.uri)

    def __delitem__(self, uri: str):
        self._remove(uri)

    def __iter__(self):
        return iter(self[i] for i in range(len(self)))

    @property
    def meta(self) -> GroupMetadata:
        """
        The metadata of the Group contained as a key-value structure.
        """
        return self._meta

    @property
    def mode() -> str:
        """
        Indictes if the Group is opened in read mode ('r') or write mode ('w').
        """
        return self._query_type_to_mode[self.query_type]
