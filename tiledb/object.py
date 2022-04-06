import tiledb.cc as lt


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

    def __init__(self, type: lt.ObjectType, uri: str):
        super().__init__(type, uri)

    @property
    def type(self) -> type:
        """
        Valid TileDB object types are Array and Group.
        """
        if self._type not in self._obj_type_to_class:
            raise KeyError(f"Unknown object type: {self._type}")
        return self._obj_type_to_class[self._type]
