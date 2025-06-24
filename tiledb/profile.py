import tiledb.libtiledb as lt


class Profile(lt.Profile):
    """
    Represents a TileDB profile.
    """

    def __init__(self, name: str = None, dir: str = None):
        """Class representing a TileDB profile.

        :param name: The name of the profile.
        :param dir: The directory of the profile.
        :raises tiledb.TileDBError:
        """
        super().__init__(name, dir)

    @property
    def name(self):
        """The name of the profile.

        :rtype: str
        """
        return self._name

    @property
    def dir(self):
        """The directory of the profile.

        :rtype: str
        """
        return self._dir

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
        :return: The parameter value.
        :rtype: str
        :raises KeyError: If the parameter does not exist.
        :raises tiledb.TileDBError:
        """
        return self.get(param, raise_keyerror=True)

    def get(self, param: str, raise_keyerror: bool = True):
        """Gets a parameter for the profile.

        :param param: The parameter name.
        :param raise_keyerror: Whether to raise a KeyError if the parameter does not exist.
        :return: The parameter value or None if it does not exist and raise_keyerror is False.
        :rtype: str or None
        :raises KeyError: If the parameter does not exist and raise_keyerror is True.
        :raises tiledb.TileDBError:
        """
        val = self._get_param(param)
        if val is None and raise_keyerror:
            raise KeyError(param)

        return val

    def save(self):
        """Saves the profile to storage.

        :raises tiledb.TileDBError:
        """
        self._save()

    @classmethod
    def load(cls, name: str = None, dir: str = None) -> "Profile":
        """Loads a profile from storage.

        :param name: The name of the profile.
        :param dir: The directory of the profile.
        :return: The loaded profile.
        :rtype: tiledb.Profile
        :raises tiledb.TileDBError:
        """
        # This is a workaround for the from_pybind11 method due to the fact
        # that this class does not inherit from CtxMixin, as is commonly done.
        lt_obj = lt.Profile._load(name, dir)
        py_obj = cls.__new__(cls)
        lt.Profile.__init__(py_obj, lt_obj)
        return py_obj

    @classmethod
    def remove(cls, name: str = None, dir: str = None):
        """Removes a profile from storage.

        :param name: The name of the profile.
        :param dir: The directory of the profile.
        :raises tiledb.TileDBError:
        """
        lt.Profile._remove(name, dir)
