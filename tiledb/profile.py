import tiledb.libtiledb as lt


class Profile(lt.Profile):
    """
    Represents a TileDB profile.
    """

    def __init__(self, name: str = None, homedir: str = None):
        """Class representing a TileDB profile.

        :param name: The name of the profile.
        :param homedir: The home directory of the profile.
        :raises tiledb.TileDBError:
        """
        super().__init__(name, homedir)

    @property
    def name(self):
        """The name of the profile.

        :rtype: str
        """
        return self._name

    @property
    def homedir(self):
        """The home directory of the profile.

        :rtype: str
        """
        return self._homedir

    def __repr__(self):
        """String representation of the profile.

        :rtype: str
        """
        return self._dump()

    def __setitem__(self, param: str, value: str):
        """Sets a parameter for the profile.

        :param param: The parameter name.
        :param value: The parameter value.
        :raises tiledb.TileDBError:
        """
        self._set_param(param, value)

    def __getitem__(self, param: str):
        """Gets a parameter for the profile.

        :param param: The parameter name.
        :raises tiledb.TileDBError:
        """
        return self._get_param(param)

    def save(self):
        """Saves the profile to storage.

        :raises tiledb.TileDBError:
        """
        self._save()

    @classmethod
    def load(cls, name: str = None, homedir: str = None) -> "Profile":
        """Loads a profile from storage.

        :param name: The name of the profile.
        :param homedir: The home directory of the profile.
        :return: The loaded profile.
        :rtype: tiledb.Profile
        :raises tiledb.TileDBError:
        """
        # This is a workaround for the from_pybind11 method due to the fact
        # that this class does not inherit from CtxMixin, as is commonly done.
        lt_obj = lt.Profile._load(name, homedir)
        py_obj = cls.__new__(cls)
        lt.Profile.__init__(py_obj, lt_obj)
        return py_obj

    def remove(self):
        """Removes the profile from storage.

        :raises tiledb.TileDBError:
        """
        self._remove()

    def dump(self):
        """Dumps the profile."""
        print(self._dump(), "\n")
