import tiledb.libtiledb as lt

from .ctx import Ctx, CtxMixin
from .domain import Domain
from .ndrectangle import NDRectangle


class CurrentDomain(CtxMixin, lt.CurrentDomain):
    """
    Represents a TileDB current domain.
    """

    def __init__(self, ctx: Ctx):
        """Class representing the current domain of a TileDB Array.

        :param ctx: A TileDB Context
        :raises tiledb.TileDBError:
        """
        super().__init__(ctx)

    @property
    def type(self):
        """The type of the current domain.

        :rtype: tiledb.CurrentDomainType
        """
        return self._type

    @property
    def is_empty(self):
        """Checks if the current domain is empty.

        :rtype: bool
        """
        return self._is_empty

    def set_ndrectangle(self, ndrect: NDRectangle):
        """Sets an N-dimensional rectangle representation on a current domain.

        :param ndrect: The N-dimensional rectangle to be used.
        :raises tiledb.TileDBError:
        """
        self._set_ndrectangle(ndrect)

    @property
    def ndrectangle(self):
        """Gets the N-dimensional rectangle associated with the current domain object.

        :rtype: NDRectangle
        :raises tiledb.TileDBError:
        """
        return NDRectangle.from_pybind11(self._ctx, self._ndrectangle())
