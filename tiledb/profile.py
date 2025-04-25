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

    def set_param(self, param: str, value: str):
        """Sets a parameter for the profile.

        :param param: The parameter name.
        :param value: The parameter value.
        :raises tiledb.TileDBError:
        """
        self._set_param(param, value)

    # maybe provide square brackets overload?
    def get_param(self, param: str):
        """Gets a parameter for the profile.

        :param param: The parameter name.
        :raises tiledb.TileDBError:
        """
        return self._get_param(param)

    def save(self):
        """Saves the profile.

        :raises tiledb.TileDBError:
        """
        self._save()

    def remove(self):
        """Removes the profile.

        :raises tiledb.TileDBError:
        """
        self._remove()

    def dump(self):
        """Dumps the profile."""
        print(self._dump(), "\n")
