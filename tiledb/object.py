from typing import Optional

import tiledb
import tiledb.cc as lt


class Object(lt.Object):
    """
    Represents a TileDB object which may be of type Array, Group, or Invalid.
    """

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
        if self._type == lt.ObjectType.ARRAY:
            return tiledb.Array
        if self._type == lt.ObjectType.GROUP:
            return tiledb.Group
        raise KeyError(f"Unknown object type: {self._type}")

    @property
    def name(self) -> Optional[str]:
        """
        :return: Name of the Object if given. Otherwise, None.
        :rtype: str
        """
        return self._name
