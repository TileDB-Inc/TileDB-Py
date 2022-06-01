import tiledb.cc as lt

from typing import Optional


class Object(lt.Object):
    """
    Represents a TileDB object which may be of type Array, Group, or Invalid.
    """

    from .libtiledb import Array
    from .group import Group

    _obj_type_to_class = {
        lt.ObjectType.ARRAY: Array,
        lt.ObjectType.GROUP: Group,
    }

    def __init__(self, type: lt.ObjectType, uri: str, name: Optional[str] = None):
        super().__init__(type, uri, name)

    @property
    def uri(self) -> str:
        """
        :return: URI of the Object.
        :rtype: str
        """
        return self._uri

    @property
    def type(self) -> type:
        """
        :return: Valid TileDB object types are Array and Group.
        :rtype: type
        """
        if self._type not in self._obj_type_to_class:
            raise KeyError(f"Unknown object type: {self._type}")
        return self._obj_type_to_class[self._type]

    @property
    def name(self) -> Optional[str]:
        """
        :return: Name of the Object if given. Otherwise, None.
        :rtype: str
        """
        return self._name
